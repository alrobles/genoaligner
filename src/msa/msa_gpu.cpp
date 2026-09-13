// GPU driver for genomsa: guide tree on host, one msa_pp_trace_kernel launch
// per guide-tree LEVEL over its independent node pairs (packed buffers),
// reference merge on host. Semantics are exactly msa_align_with_tree's --
// the level-batched decomposition is gated bit-exact by
// tests/parity/msa_pipeline_parity.cpp.
//
// This file needs a HIP compiler (hipcc on AMD, nvcc -x cu on NVIDIA); it is
// NOT part of the CPU shim path. The kernel body it launches is the shipped
// one, identical source the shim tests replay.

#include <genoaligner/msa/msa.hpp>
#include <genoaligner/backend/msa_pp_kernel_impl.hip>
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef GENOALIGNER_HIP_SHIM
// Launch-owner definitions for the CPU shim (same arrangement as
// src/api/api.cpp): the shim declares these extern; this TU owns them.
thread_local uint3 threadIdx{0, 0, 0};
uint3 blockIdx{0, 0, 0};
dim3  blockDim{1, 1, 1};
dim3  gridDim{1, 1, 1};
#endif

namespace genomsa {

using genoaligner::MSA_PP_TOO_BIG;
using genoaligner::MsaPPPair;
using genoaligner::MsaPPParams;
using genoaligner::MsaPPResult;
using genoaligner::MsaProfileView;
using genoaligner::msa_pp_trace_kernel;
using genoaligner::msa_pp_trace_kernel_wf;

// V2 wavefront launch width: multiple of both warp sizes (gfx 64, NV 32).
static constexpr int WF_T = 256;

static void hip_free_all(std::vector<void*>& ps) {
    for (void* p : ps) if (p) hipFree(p);
    ps.clear();
}

bool msa_align_gpu(const std::vector<std::string>& seqs, const Params& P,
                   std::vector<std::string>& out, std::string& err,
                   GpuStats* stats, Tree* guide_out) {
    auto fail = [&](const char* m) { err = m; return false; };
    const int n = (int)seqs.size();
    if (n == 0) return fail("empty input");
    if (n == 1) { out = seqs; return true; }

    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        return fail("no HIP device");

    auto t0 = std::chrono::steady_clock::now();
    // Host-side stages run multithreaded; both _mt variants are bit-exact
    // with the sequential reference semantics (gate: driver parity).
    const int nthreads = [] {
        const char* e = std::getenv("GENOMSA_THREADS");
        int t = e ? std::atoi(e) : (int)std::thread::hardware_concurrency();
        return t > 0 ? t : 1;
    }();
    std::vector<float> D = kmer_distances_mt(seqs, P.kmer_k, nthreads);
    auto t1 = std::chrono::steady_clock::now();
    Tree tree = nj_tree_mt(D, n, nthreads);
    if (guide_out) *guide_out = tree;
    auto levels = tree_levels(tree);
    auto t2 = std::chrono::steady_clock::now();

    std::vector<Profile> profs(tree.nodes.size());
    for (int i = 0; i < n; ++i) {
        profs[i] = profile_from_seq(seqs[i]);
        profs[i].ids[0] = i;
    }
    const MsaPPParams kp{P.match, P.ts, P.tv, P.gap_open, P.gap_extend,
                         P.free_end_gaps ? 1 : 0};

    std::vector<void*> dev;
    for (const auto& level : levels) {
        const int np = (int)level.size();
        // ---- pack this level's pairs -----------------------------------
        std::vector<MsaPPPair> hp(np);
        std::vector<size_t> dir_base(np), dir_av(np),
                            scr_base(np), scr_av(np), cig_base(np);
        std::vector<float> fdata;                   // all column data, packed
        size_t dtot = 0, stot = 0, ctot = 0;
        int cap = 0;
        for (int k = 0; k < np; ++k) {
            const Profile& A = profs[tree.nodes[level[k]].left];
            const Profile& B = profs[tree.nodes[level[k]].right];
            hp[k].A.len = A.ncols(); hp[k].B.len = B.ncols();
            dir_base[k] = dtot;
            dir_av[k]   = (size_t)(A.ncols() + 1) * (B.ncols() + 1);
            dtot += dir_av[k];
            scr_base[k] = stot;
            // serial needs 3*(N+1)+2*(M+1); wavefront needs 11*(M+1)+2*(N+1)
            const size_t s_ser = (size_t)3 * (B.ncols() + 1) + 2 * (A.ncols() + 1);
            const size_t s_wf  = (size_t)11 * (A.ncols() + 1) + 2 * (B.ncols() + 1);
            scr_av[k]   = s_ser > s_wf ? s_ser : s_wf;
            stot += scr_av[k];
            cig_base[k] = ctot;
            int ck = A.ncols() + B.ncols() + 4;
            if (ck > cap) cap = ck;
            ctot += (size_t)ck;
            for (int side = 0; side < 2; ++side) {
                const Profile& S = side ? B : A;
                MsaProfileView& v = side ? hp[k].B : hp[k].A;
                // record byte offsets now; patch to device pointers below
                for (int b = 0; b < 5; ++b) {
                    v.cols[b] = (const float*)(size_t)fdata.size();
                    for (const auto& c : S.cols) fdata.push_back(c[b]);
                }
                v.occ = (const float*)(size_t)fdata.size();
                for (float o : S.occ) fdata.push_back(o);
            }
        }

        // ---- device buffers ---------------------------------------------
        float*      d_fdata = nullptr;
        MsaPPPair*  d_pairs = nullptr;
        MsaPPResult* d_res  = nullptr;
        uint8_t*    d_dirs  = nullptr;
        uint8_t*    d_cig   = nullptr;
        int*        d_meta  = nullptr;
        float*      d_scr   = nullptr;
        size_t*     d_db = nullptr; size_t* d_da = nullptr;
        size_t*     d_cb = nullptr;
        size_t*     d_sb = nullptr; size_t* d_sa = nullptr;
        auto mal = [&](void** p, size_t bytes, const char* what) {
            if (bytes == 0) bytes = 8;
            if (hipMalloc(p, bytes) != hipSuccess) {
                err = std::string("hipMalloc ") + what;
                return false;
            }
            dev.push_back(*p);
            return true;
        };
        if (!mal((void**)&d_fdata, fdata.size() * sizeof(float), "fdata") ||
            !mal((void**)&d_pairs, np * sizeof(MsaPPPair), "pairs") ||
            !mal((void**)&d_res,   np * sizeof(MsaPPResult), "res") ||
            !mal((void**)&d_dirs,  dtot, "dirs") ||
            !mal((void**)&d_cig,   ctot, "cig") ||
            !mal((void**)&d_meta,  np * 2 * sizeof(int), "meta") ||
            !mal((void**)&d_scr,   stot * sizeof(float), "scratch") ||
            !mal((void**)&d_db, np * sizeof(size_t), "dir_base") ||
            !mal((void**)&d_da, np * sizeof(size_t), "dir_avail") ||
            !mal((void**)&d_cb, np * sizeof(size_t), "cig_base") ||
            !mal((void**)&d_sb, np * sizeof(size_t), "scr_base") ||
            !mal((void**)&d_sa, np * sizeof(size_t), "scr_avail")) {
            hip_free_all(dev);
            return false;
        }
        if (stats && dtot > stats->dir_bytes) stats->dir_bytes = dtot;

        // patch the packed offsets into device pointers
        for (int k = 0; k < np; ++k) {
            for (int side = 0; side < 2; ++side) {
                MsaProfileView& v = side ? hp[k].B : hp[k].A;
                for (int b = 0; b < 5; ++b)
                    v.cols[b] = d_fdata + (size_t)v.cols[b];
                v.occ = d_fdata + (size_t)v.occ;
            }
        }

        auto cp = [&](void* d, const void* s, size_t bytes, const char* what,
                      hipMemcpyKind kind = hipMemcpyHostToDevice) {
            if (bytes && hipMemcpy(d, s, bytes, kind) != hipSuccess) {
                err = std::string("hipMemcpy ") + what;
                return false;
            }
            return true;
        };
        if (!cp(d_fdata, fdata.data(), fdata.size() * sizeof(float), "fdata") ||
            !cp(d_pairs, hp.data(), np * sizeof(MsaPPPair), "pairs") ||
            !cp(d_db, dir_base.data(), np * sizeof(size_t), "dir_base") ||
            !cp(d_da, dir_av.data(),   np * sizeof(size_t), "dir_avail") ||
            !cp(d_cb, cig_base.data(), np * sizeof(size_t), "cig_base") ||
            !cp(d_sb, scr_base.data(), np * sizeof(size_t), "scr_base") ||
            !cp(d_sa, scr_av.data(),   np * sizeof(size_t), "scr_avail")) {
            hip_free_all(dev);
            return false;
        }

        // Kernel selection: wavefront (block-per-pair) is the default;
        // GENOMSA_MSA_KERNEL=serial keeps the one-thread-per-pair kernel for
        // A/B benchmarking on device. Both are parity-gated.
        static const bool use_wf = [] {
            const char* k = std::getenv("GENOMSA_MSA_KERNEL");
            return !(k && std::strcmp(k, "serial") == 0);
        }();
#ifdef GENOALIGNER_HIP_SHIM
        if (use_wf) {
            // Genuine threaded emulation: run_block spawns WF_T real host
            // threads per pair with a real block barrier for __syncthreads.
            gridDim  = dim3{(unsigned)np, 1, 1};
            blockDim = dim3{WF_T, 1, 1};
            for (int bx = 0; bx < np; ++bx) {
                blockIdx = uint3{(unsigned)bx, 0, 0};
                shim::run_block(WF_T, [&] {
                    msa_pp_trace_kernel_wf(d_pairs, kp, d_res, d_dirs,
                                           d_db, d_da, d_cig, d_cb, d_meta,
                                           d_scr, d_sb, d_sa, np, cap);
                });
            }
        } else {
            blockDim = dim3{128, 1, 1};
            gridDim  = dim3{(unsigned)((np + 127) / 128), 1, 1};
            for (unsigned bx = 0; bx < gridDim.x; ++bx) {
                blockIdx = uint3{bx, 0, 0};
                for (unsigned tx = 0; tx < 128; ++tx) {
                    if (bx * 128 + tx >= (unsigned)np) break;
                    threadIdx = uint3{tx, 0, 0};
                    msa_pp_trace_kernel(d_pairs, kp, d_res, d_dirs, d_db, d_da,
                                        d_cig, d_cb, d_meta, d_scr, d_sb, d_sa,
                                        np, cap);
                }
            }
        }
        hipError_t le = hipSuccess, se = hipSuccess;
#else
        if (use_wf) {
            hipLaunchKernelGGL(msa_pp_trace_kernel_wf,
                               dim3((unsigned)np), dim3(WF_T), 0, nullptr,
                               d_pairs, kp, d_res, d_dirs, d_db, d_da,
                               d_cig, d_cb, d_meta, d_scr, d_sb, d_sa,
                               np, cap);
        } else {
            hipLaunchKernelGGL(msa_pp_trace_kernel,
                               dim3((unsigned)((np + 127) / 128)), dim3(128),
                               0, nullptr,
                               d_pairs, kp, d_res, d_dirs, d_db, d_da,
                               d_cig, d_cb, d_meta, d_scr, d_sb, d_sa,
                               np, cap);
        }
        hipError_t le = hipGetLastError();
        hipError_t se = hipDeviceSynchronize();
#endif
        if (le != hipSuccess || se != hipSuccess) {
            err = std::string("kernel: ") + hipGetErrorString(le != hipSuccess ? le : se);
            hip_free_all(dev);
            return false;
        }

        std::vector<MsaPPResult> hres(np);
        std::vector<int>         hmeta(np * 2);
        std::vector<uint8_t>     hcig(ctot);
        if (!cp(hres.data(), d_res, np * sizeof(MsaPPResult), "res",
                hipMemcpyDeviceToHost) ||
            !cp(hmeta.data(), d_meta, np * 2 * sizeof(int), "meta",
                hipMemcpyDeviceToHost) ||
            !cp(hcig.data(),  d_cig, ctot, "cig", hipMemcpyDeviceToHost)) {
            hip_free_all(dev);
            return false;
        }
        hip_free_all(dev);

        static const char ops[] = "MID";
        for (int k = 0; k < np; ++k) {
            const int u = level[k];
            if (hres[k].score == MSA_PP_TOO_BIG) {
                err = "pair matrix exceeded packed dir workspace";
                return false;
            }
            AlignResult r;
            r.score = hres[k].score;
            r.ai = hres[k].ai; r.aj = hres[k].aj;
            r.bi = hres[k].bi; r.bj = hres[k].bj;
            const uint8_t* cb = hcig.data() + cig_base[k];
            for (int q = hmeta[k * 2 + 1] - 1; q >= 0; --q)
                r.cigar.push_back(ops[cb[q]]);
            const auto& nd = tree.nodes[u];
            profs[u] = merge_profiles(profs[nd.left], profs[nd.right], r);
        }
        if (stats) stats->pairs += np;
    }
    auto t3 = std::chrono::steady_clock::now();

    out.assign(n, std::string());
    const Profile& root = profs[tree.root];
    for (int k = 0; k < n; ++k) out[root.ids[k]] = root.rows[k];
    if (stats) {
        stats->levels = (int)levels.size();
        stats->dist_s  = std::chrono::duration<double>(t1 - t0).count();
        stats->tree_s  = std::chrono::duration<double>(t2 - t1).count();
        stats->align_s = std::chrono::duration<double>(t3 - t2).count();
    }
    return true;
}

} // namespace genomsa
