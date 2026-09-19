// Pipeline parity: the LEVEL-BATCHED driver structure (the shape the GPU
// path will take -- independent nodes per level -> kernel batch -> merge)
// must produce output IDENTICAL to the sequential reference driver.
//
// For every level: each node is aligned by the shipped msa_pp_trace_kernel
// body (one direct call per pair under the CPU shim, exactly what one thread
// per pair does on device), merged by the reference merge, and the next
// level proceeds. If the final MSA differs from msa_align_with_tree on the
// same guide tree, the batch decomposition -- not just the kernel -- is
// wrong, and that is the failure mode this gate exists to catch.
//
// Build:  g++ -O2 -std=c++17 -DGENOALIGNER_HIP_SHIM \
//           -I tests/parity/hip_cpu_shim -I include \
//           tests/parity/msa_pipeline_parity.cpp src/msa/msa_ref.cpp
#include <genoaligner/msa/msa.hpp>
#include <cstdio>
#include <random>
#include <vector>
#include <string>

#include <hip/hip_runtime.h>
#include <genoaligner/backend/msa_pp_kernel_impl.hip>

thread_local uint3 threadIdx{0,0,0};
uint3 blockIdx{0,0,0};
dim3  blockDim{1,1,1};
dim3  gridDim{1,1,1};

using namespace genoaligner;
using genomsa::Profile;
using genomsa::Params;
using genomsa::Tree;

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); ++fails; } } while (0)

struct SoAHolder {
    std::vector<float> c[MSA_MAX_SYMS];
    std::vector<float> occ;
    MsaProfileView view;
};
static SoAHolder to_soa(const Profile& p) {
    SoAHolder h;
    h.occ = p.occ;
    for (int b = 0; b < p.alpha; ++b) {
        h.c[b].resize(p.cols.size());
        for (size_t i = 0; i < p.cols.size(); ++i) h.c[b][i] = p.cols[i][b];
        h.view.cols[b] = h.c[b].data();
    }
    h.view.occ = h.occ.data();
    h.view.len = p.ncols();
    return h;
}
static MsaPPParams to_params(const Params& P) {
    MsaPPParams k{P.match, P.ts, P.tv, P.gap_open, P.gap_extend,
                  P.free_end_gaps ? 1 : 0,
                  P.psgp ? 1 : 0, P.psgp_scale, P.psgp_min_open,
                  P.psgp_min_ext, P.alpha, {}, nullptr};
    if (P.alpha > 4 && P.alpha <= MSA_SUB_INLINE_SYMS)
        memcpy(k.sub, P.sub.data(), sizeof(k.sub));
    if (P.alpha > MSA_SUB_INLINE_SYMS)
        k.sub_ext = P.sub.data();   // host pointer under the CPU shim
    return k;
}

// One node of one level, through the kernel body (the GPU thread's work).
static genomsa::AlignResult kernel_align(const Profile& A, const Profile& B,
                                         const Params& P) {
    SoAHolder ha = to_soa(A), hb = to_soa(B);
    MsaPPPair pp{ha.view, hb.view};
    const size_t dir_sz = (size_t)(A.ncols() + 1) * (B.ncols() + 1);
    const size_t scr_sz = (size_t)3 * (B.ncols() + 1) + 2 * (A.ncols() + 1);
    const int    cap    = A.ncols() + B.ncols() + 4;
    std::vector<uint8_t> dirs(dir_sz), cig(cap);
    std::vector<float>   scr(scr_sz);
    int meta[2];
    MsaPPResult res;
    const size_t zero[1] = {0}, av_d[1] = {dir_sz}, av_s[1] = {scr_sz};
    blockIdx = uint3{0,0,0}; threadIdx = uint3{0,0,0};
    blockDim = dim3{1,1,1};  gridDim  = dim3{1,1,1};
    msa_pp_trace_kernel(&pp, to_params(P), &res, dirs.data(), zero, av_d,
                        cig.data(), zero, meta, scr.data(), zero, av_s, 1, cap);
    genomsa::AlignResult r;
    r.score = res.score; r.ai = res.ai; r.aj = res.aj; r.bi = res.bi; r.bj = res.bj;
    static const char ops[] = "MID";
    for (int k = meta[1] - 1; k >= 0; --k) r.cigar.push_back(ops[cig[k]]);
    return r;
}

// The level-batched pipeline: identical in structure to a GPU driver that
// launches one kernel per level over its independent node pairs.
static std::vector<std::string> msa_align_batched(
        const std::vector<std::string>& seqs, const Tree& tree,
        const Params& P) {
    const int n = (int)seqs.size();
    std::vector<Profile> profs(tree.nodes.size());
    for (int i = 0; i < n; ++i) {
        profs[i] = genomsa::profile_from_seq(seqs[i]);
        profs[i].ids[0] = i;
    }
    for (const auto& level : genomsa::tree_levels(tree)) {
        for (int u : level) {                    // independent: one GPU launch
            const auto& nd = tree.nodes[u];
            genomsa::AlignResult r;
            if (P.gappy > 0.0f) {          // same strip/expand as the driver
                genomsa::GappyStrip sa =
                    genomsa::profile_strip(profs[nd.left],  P.gappy);
                genomsa::GappyStrip sb =
                    genomsa::profile_strip(profs[nd.right], P.gappy);
                r = genomsa::cigar_expand_gappy(
                    kernel_align(sa.prof, sb.prof, P), sa, sb, P);
            } else {
                r = kernel_align(profs[nd.left], profs[nd.right], P);
            }
            profs[u] = genomsa::merge_profiles(profs[nd.left], profs[nd.right], r);
        }
    }
    std::vector<std::string> out(n);
    const Profile& root = profs[tree.root];
    for (int k = 0; k < n; ++k) out[root.ids[k]] = root.rows[k];
    return out;
}

static std::mt19937 rng(777);
static std::string rand_dna(int len) {
    static const char b[] = "ACGT";
    std::string s; for (int i = 0; i < len; ++i) s += b[rng() % 4]; return s;
}
static std::string mutate(const std::string& s, int ps, int pi, int pd) {
    static const char b[] = "ACGT";
    std::string o;
    for (char c : s) {
        int r = rng() % 100;
        if (r < pd) continue;
        o += (r < pd + ps) ? b[rng() % 4] : c;
        if (rng() % 100 < pi) o += b[rng() % 4];
    }
    return o;
}

static void run_suite(Params P, const char* tag0) {
    // related families, several sizes incl. non-powers of two and a
    // degenerate n=2 (single level, single node)
    for (int n : {2, 3, 4, 5, 7, 8, 11, 16}) {
        std::string anc = rand_dna(60 + rng() % 80);
        std::vector<std::string> in;
        for (int i = 0; i < n; ++i) in.push_back(mutate(anc, 12, 6, 6));
        // add a fragment and an ambiguous-base record sometimes
        if (n >= 5) in[3] = in[3].substr(4, in[3].size() / 2);
        if (n >= 7) { std::string& q = in[6]; if (q.size() > 10) q[5] = 'N'; }

        std::vector<float> D = genomsa::kmer_distances(in, P.kmer_k);
        Tree t = genomsa::nj_tree(D, n);
        auto ref = genomsa::msa_align_with_tree(in, t, P);
        auto bat = msa_align_batched(in, t, P);
        CHECK(bat == ref, "%sn=%d batched != sequential", tag0, n);
        // invariants on the batched output too
        for (int i = 0; i < n; ++i) {
            std::string u = bat[i];
            u.erase(std::remove(u.begin(), u.end(), '-'), u.end());
            CHECK(u == in[i], "%sn=%d seq %d corrupted", tag0, n, i);
        }
    }
}

int main() {
    run_suite(Params{}, "default:");           // psgp + gappy(0.95) on
    Params l; l.psgp = false; l.gappy = 0;
    run_suite(l, "legacy:");
    Params p; p.gappy = 0;
    run_suite(p, "psgp:");
    Params g; g.psgp = false; g.gappy = 0.9f;  // strip runs >90% gap
    run_suite(g, "gappy:");
    Params pg; pg.gappy = 0.9f;
    run_suite(pg, "psgp+gappy:");

    if (!fails) printf("RESULT: PASS -- level-batched kernel pipeline == sequential reference\n");
    else        printf("%d FAILURES\n", fails);
    return fails;
}
