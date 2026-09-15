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
#define GENOALIGNER_CODON_REFINE_DEF
#include <genoaligner/backend/codon_refine_kernel_impl.hip>
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
    std::vector<float> D = kmer_distances_mt(seqs, P.kmer_k, nthreads, P.alpha);
    auto t1 = std::chrono::steady_clock::now();
    // Guide tree: NJ0 on the device unless GENOMSA_NJ=cpu. nj_tree_gpu is
    // bit-exact with nj_tree, so the fallback below changes timing only.
    Tree tree;
    {
        const char* e = std::getenv("GENOMSA_NJ");
        const bool want_gpu = !(e && std::strcmp(e, "cpu") == 0);
        std::string nj_err;
        if (want_gpu) tree = nj_tree_gpu(D, n, nj_err);
        if (!want_gpu || !nj_err.empty()) {
            if (!nj_err.empty())
                fprintf(stderr, "genomsa: %s; falling back to nj_tree_mt\n", nj_err.c_str());
            tree = nj_tree_mt(D, n, nthreads);
        } else if (stats) {
            stats->tree_gpu = 1;
        }
    }
    if (guide_out) *guide_out = tree;
    auto levels = tree_levels(tree);
    auto t2 = std::chrono::steady_clock::now();

    std::vector<Profile> profs(tree.nodes.size());
    for (int i = 0; i < n; ++i) {
        profs[i] = profile_from_seq(seqs[i], P.alpha);
        profs[i].ids[0] = i;
    }
    MsaPPParams kp{P.match, P.ts, P.tv, P.gap_open, P.gap_extend,
                   P.free_end_gaps ? 1 : 0,
                   P.psgp ? 1 : 0, P.psgp_scale, P.psgp_min_open,
                   P.psgp_min_ext, P.alpha, {}, nullptr};
    if (P.alpha > 4 && P.alpha <= MSA_SUB_INLINE_SYMS)
        std::memcpy(kp.sub, P.sub.data(), sizeof(kp.sub));
    // alpha > 20 (codon mode): the 65x65 matrix exceeds the kernel-param
    // space, so it lives in a device buffer allocated once per call.
    float* d_sub = nullptr;
    if (P.alpha > MSA_SUB_INLINE_SYMS) {
        const size_t bytes = (size_t)P.alpha * P.alpha * sizeof(float);
        if (hipMalloc((void**)&d_sub, bytes) != hipSuccess ||
            hipMemcpy(d_sub, P.sub.data(), bytes, hipMemcpyHostToDevice)
                != hipSuccess) {
            return fail("codon substitution matrix upload");
        }
        kp.sub_ext = d_sub;
    }
    // freed on every exit path, success or failure
    struct SubGuard { float* p; ~SubGuard() { if (p) hipFree(p); } }
        sub_guard{d_sub};

    static const bool lvldbg = std::getenv("GENOMSA_LVL_DEBUG") != nullptr;
    auto fnv = [](const void* p, size_t n, uint64_t h) {
        const uint8_t* b = (const uint8_t*)p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
        return h;
    };
    auto prof_hash = [&](const Profile& p) {
        uint64_t h = 1469598103934665603ull;
        for (const auto& r : p.rows) h = fnv(r.data(), r.size(), h);
        h = fnv(p.cols.data(), p.cols.size() * sizeof(p.cols[0]), h);
        h = fnv(p.occ.data(), p.occ.size() * sizeof(float), h);
        return h;
    };
    int lvli = 0;

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
        // gappy-column heuristic: strip before packing, expand after the
        // kernel (same strip as the reference -> parity preserved)
        std::vector<GappyStrip> strips_a(np), strips_b(np);
        std::vector<Profile>    pa(np), pb(np);
        for (int k = 0; k < np; ++k) {
            if (P.gappy > 0.0f) {
                strips_a[k] = profile_strip(profs[tree.nodes[level[k]].left],  P.gappy);
                strips_b[k] = profile_strip(profs[tree.nodes[level[k]].right], P.gappy);
                pa[k] = strips_a[k].prof;
                pb[k] = strips_b[k].prof;
            }
        }
        for (int k = 0; k < np; ++k) {
            const Profile& A = P.gappy > 0.0f ? pa[k] : profs[tree.nodes[level[k]].left];
            const Profile& B = P.gappy > 0.0f ? pb[k] : profs[tree.nodes[level[k]].right];
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
                for (int b = 0; b < P.alpha; ++b) {
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
        if (lvldbg) {
            uint64_t h = fnv(fdata.data(), fdata.size() * sizeof(float),
                             1469598103934665603ull);
            fprintf(stderr, "LVL %d np=%d fdata=%016zx\n", lvli, np,
                    (size_t)h);
            for (int k = 0; k < np; ++k)
                fprintf(stderr, "  in k=%d u=%d A=%016zx B=%016zx\n", k,
                        level[k],
                        (size_t)prof_hash(P.gappy > 0.0f ? pa[k]
                                : profs[tree.nodes[level[k]].left]),
                        (size_t)prof_hash(P.gappy > 0.0f ? pb[k]
                                : profs[tree.nodes[level[k]].right]));
        }

        // patch the packed offsets into device pointers
        for (int k = 0; k < np; ++k) {
            for (int side = 0; side < 2; ++side) {
                MsaProfileView& v = side ? hp[k].B : hp[k].A;
                for (int b = 0; b < P.alpha; ++b)
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
        if (lvldbg) {
            uint64_t h = fnv(hres.data(), np * sizeof(MsaPPResult),
                             1469598103934665603ull);
            h = fnv(hmeta.data(), np * 2 * sizeof(int), h);
            h = fnv(hcig.data(), ctot, h);
            fprintf(stderr, "LVL %d out=%016zx\n", lvli, (size_t)h);
        }

        static const char ops[] = "MID";
        static const bool dbg = std::getenv("GENOMSA_PAIR_DEBUG") != nullptr;
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
            if (dbg) {
                const Profile& A = P.gappy > 0.0f ? pa[k]
                                   : profs[tree.nodes[level[k]].left];
                const Profile& B = P.gappy > 0.0f ? pb[k]
                                   : profs[tree.nodes[level[k]].right];
                AlignResult ref = align_profiles(A, B, P);
                if (ref.cigar != r.cigar || ref.score != r.score ||
                    ref.ai != r.ai || ref.aj != r.aj ||
                    ref.bi != r.bi || ref.bj != r.bj)
                    fprintf(stderr,
                            "PAIRDBG node=%d k=%d M=%d N=%d | kscore=%.6g refscore=%.6g "
                            "kcig=%.60s refcig=%.60s | k.ai=%d r.ai=%d k.bi=%d r.bi=%d\n",
                            u, k, A.ncols(), B.ncols(), r.score, ref.score,
                            r.cigar.c_str(), ref.cigar.c_str(),
                            r.ai, ref.ai, r.bi, ref.bi);
            }
            if (P.gappy > 0.0f)
                r = cigar_expand_gappy(r, strips_a[k], strips_b[k], P);
            const auto& nd = tree.nodes[u];
            profs[u] = merge_profiles(profs[nd.left], profs[nd.right], r);
            if (lvldbg)
                fprintf(stderr, "  prof u=%d h=%016zx\n", u,
                        (size_t)prof_hash(profs[u]));
        }
        if (stats) stats->pairs += np;
        ++lvli;
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

// ---------------------------------------------------------------------------
// Codon stage-2 on device: per round, the host rebuilds the codon-column
// profile + dot table, packs per-row footprints, launches
// codon_refine_kernel (one thread per row: phase DP -> tr bytes + endpoint),
// then traces back + merges on host (genomsa::detail::, shared with
// codon_refine). Under GENOALIGNER_HIP_SHIM the kernel body runs per-thread
// on CPU -- the shim binary's --cpu vs default refine outputs must be
// byte-identical.
// ---------------------------------------------------------------------------
bool codon_refine_gpu(const std::vector<std::string>& seqs_nt,
                      const std::vector<std::string>& aln_nt,
                      const Params& P, int rounds,
                      std::vector<std::string>& out, std::string& err,
                      float* kernel_s) {
    auto fail = [&](const char* m) { err = m; return false; };
    if (rounds < 1 || aln_nt.empty()) { out = aln_nt; return true; }
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        return fail("no HIP device");

    const int n = (int)aln_nt.size();
    const int D = (int)P.codon_refine_band;
    const int ND = 2 * D + 1;
    std::vector<std::string> cur = aln_nt;
    std::vector<int> cofs(cur[0].size() / 3);
    for (size_t j = 0; j < cofs.size(); ++j) cofs[j] = 3 * (int)j;
    double kt = 0.0;

    for (int r = 0; r < rounds; ++r) {
        detail::CodonProf cp = detail::codon_prof_build(cur, cofs);
        const int C = cp.C;
        std::vector<float> dot((size_t)65 * C, 0.f);
        for (int t = 0; t < 65; ++t)
            for (int j = 0; j < C; ++j) {
                float s = 0;
                for (int b = 0; b < 65; ++b)
                    if (cp.ccnt[j][b]) s += cp.ccnt[j][b] * P.sub[t * 65 + b];
                dot[(size_t)t * C + j] = s;
            }

        // ---- pack per-row inputs ---------------------------------------
        std::string seqbuf;
        std::vector<size_t> seq_base(n), pi_base(n), tr_base(n), scr_base(n);
        std::vector<int>    seq_len(n), pi_len(n);
        std::vector<int8_t> stok((size_t)n * C);
        std::vector<int>    pibuf;
        size_t trtot = 0;
        for (int s = 0; s < n; ++s) {
            seq_base[s] = seqbuf.size();
            seqbuf += seqs_nt[s];
            seq_len[s] = (int)seqs_nt[s].size();
            pi_base[s] = pibuf.size();
            int K = 0;
            for (int j = 0; j < C; ++j) {
                int t = cp.tok[s][j];
                stok[(size_t)s * C + j] = (int8_t)t;
                if (t >= 0) { pibuf.push_back(j); ++K; }
            }
            pi_len[s] = K;
            tr_base[s] = trtot;
            scr_base[s] = (size_t)s * 2 * ND;
            trtot += (size_t)K * ND;
        }
        // tr is direction bytes; cap the workspace at 4 GB (beyond that the
        // host path is cheaper than the transfer anyway).
        if (trtot > (size_t)4 << 30) return fail("refine workspace > 4 GB");
        std::vector<float> ncnt((size_t)C * 12);
        for (int j = 0; j < C; ++j)
            std::memcpy(&ncnt[(size_t)j * 12], &cp.ncnt[j][0][0],
                        12 * sizeof(float));

        genoaligner::CodonRefineParams crp{P.match, P.ts, P.tv, P.gap_open,
                              P.psgp ? 1 : 0, P.psgp_scale, P.psgp_min_open,
                              P.codon_fs, P.codon_fs_term, D, n, C};

        // ---- device buffers ---------------------------------------------
        char*    d_seqs = nullptr;  size_t* d_sb = nullptr;
        int*     d_sl = nullptr;    int8_t* d_stok = nullptr;
        int*     d_pi = nullptr;    size_t* d_pb = nullptr;
        int*     d_pl = nullptr;    float* d_dot = nullptr;
        float*   d_oc = nullptr;    float* d_nc = nullptr;
        float*   d_sub = nullptr;   uint8_t* d_tr = nullptr;
        size_t*  d_tb = nullptr;    float* d_scr = nullptr;
        size_t*  d_cb = nullptr;    int* d_bes = nullptr;
        std::vector<void*> dev;
        auto mal = [&](void** p, size_t bytes, const char* what) {
            if (bytes == 0) bytes = 8;
            if (hipMalloc(p, bytes) != hipSuccess) {
                err = std::string("hipMalloc ") + what;
                return false;
            }
            dev.push_back(*p);
            return true;
        };
        auto cp2 = [&](void* d, const void* s, size_t bytes, const char* what,
                       hipMemcpyKind k = hipMemcpyHostToDevice) {
            if (bytes && hipMemcpy(d, s, bytes, k) != hipSuccess) {
                err = std::string("hipMemcpy ") + what;
                return false;
            }
            return true;
        };
        if (!mal((void**)&d_seqs, seqbuf.size(), "seqs") ||
            !mal((void**)&d_sb,   n * sizeof(size_t), "seq_base") ||
            !mal((void**)&d_sl,   n * sizeof(int), "seq_len") ||
            !mal((void**)&d_stok, stok.size(), "stok") ||
            !mal((void**)&d_pi,   pibuf.size() * sizeof(int), "pi") ||
            !mal((void**)&d_pb,   n * sizeof(size_t), "pi_base") ||
            !mal((void**)&d_pl,   n * sizeof(int), "pi_len") ||
            !mal((void**)&d_dot,  dot.size() * sizeof(float), "dot") ||
            !mal((void**)&d_oc,   C * sizeof(float), "ocnt") ||
            !mal((void**)&d_nc,   ncnt.size() * sizeof(float), "ncnt") ||
            !mal((void**)&d_sub,  65 * 65 * sizeof(float), "sub") ||
            !mal((void**)&d_tr,   trtot, "tr") ||
            !mal((void**)&d_tb,   n * sizeof(size_t), "tr_base") ||
            !mal((void**)&d_scr,  (size_t)n * 2 * ND * sizeof(float), "scr") ||
            !mal((void**)&d_cb,   n * sizeof(size_t), "scr_base") ||
            !mal((void**)&d_bes,  n * sizeof(int), "bes")) {
            hip_free_all(dev);
            return false;
        }
        if (!cp2(d_seqs, seqbuf.data(), seqbuf.size(), "seqs") ||
            !cp2(d_sb, seq_base.data(), n * sizeof(size_t), "seq_base") ||
            !cp2(d_sl, seq_len.data(), n * sizeof(int), "seq_len") ||
            !cp2(d_stok, stok.data(), stok.size(), "stok") ||
            !cp2(d_pi, pibuf.data(), pibuf.size() * sizeof(int), "pi") ||
            !cp2(d_pb, pi_base.data(), n * sizeof(size_t), "pi_base") ||
            !cp2(d_pl, pi_len.data(), n * sizeof(int), "pi_len") ||
            !cp2(d_dot, dot.data(), dot.size() * sizeof(float), "dot") ||
            !cp2(d_oc, cp.ocnt.data(), C * sizeof(float), "ocnt") ||
            !cp2(d_nc, ncnt.data(), ncnt.size() * sizeof(float), "ncnt") ||
            !cp2(d_sub, P.sub.data(), 65 * 65 * sizeof(float), "sub") ||
            !cp2(d_tb, tr_base.data(), n * sizeof(size_t), "tr_base") ||
            !cp2(d_cb, scr_base.data(), n * sizeof(size_t), "scr_base")) {
            hip_free_all(dev);
            return false;
        }

        auto k0 = std::chrono::steady_clock::now();
#ifdef GENOALIGNER_HIP_SHIM
        {
            const unsigned T = 128;
            gridDim  = dim3{(unsigned)((n + T - 1) / T), 1, 1};
            blockDim = dim3{T, 1, 1};
            for (unsigned bx = 0; bx < gridDim.x; ++bx) {
                blockIdx = uint3{bx, 0, 0};
                for (unsigned tx = 0; tx < T; ++tx) {
                    if (bx * T + tx >= (unsigned)n) break;
                    threadIdx = uint3{tx, 0, 0};
                    genoaligner::codon_refine_kernel(
                        d_seqs, d_sb, d_sl, d_stok, d_pi, d_pb, d_pl,
                        d_dot, d_oc, d_nc, d_sub, crp, d_tr, d_tb,
                        d_scr, d_cb, d_bes, n);
                }
            }
        }
        hipError_t le = hipSuccess, se = hipSuccess;
#else
        hipLaunchKernelGGL(genoaligner::codon_refine_kernel,
                           dim3((unsigned)((n + 127) / 128)), dim3(128),
                           0, nullptr,
                           d_seqs, d_sb, d_sl, d_stok, d_pi, d_pb, d_pl,
                           d_dot, d_oc, d_nc, d_sub, crp, d_tr, d_tb,
                           d_scr, d_cb, d_bes, n);
        hipError_t le = hipGetLastError();
        hipError_t se = hipDeviceSynchronize();
#endif
        kt += std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - k0).count();
        if (le != hipSuccess || se != hipSuccess) {
            err = std::string("refine kernel: ") +
                  hipGetErrorString(le != hipSuccess ? le : se);
            hip_free_all(dev);
            return false;
        }

        std::vector<int> hbes(n);
        std::vector<uint8_t> htr(trtot);
        if (!cp2(hbes.data(), d_bes, n * sizeof(int), "bes",
                 hipMemcpyDeviceToHost) ||
            !cp2(htr.data(), d_tr, trtot, "tr", hipMemcpyDeviceToHost)) {
            hip_free_all(dev);
            return false;
        }
        hip_free_all(dev);

        // ---- host traceback + merge (shared with codon_refine) ----------
        std::vector<detail::Place> pls(n);
        for (int s = 0; s < n; ++s) {
            const int K = pi_len[s];
            std::vector<int> pi(pibuf.begin() + pi_base[s],
                                pibuf.begin() + pi_base[s] + K);
            pls[s] = detail::codon_trace_place(
                seqs_nt[s], cp, s, P, pi,
                K ? htr.data() + tr_base[s] : nullptr, ND, hbes[s]);
        }
        cur = detail::codon_refine_merge(pls, C, &cofs);
    }
    if (kernel_s) *kernel_s = (float)kt;
    out = cur;
    return true;
}

} // namespace genomsa
