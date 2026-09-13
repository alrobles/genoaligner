// Parity gate: msa_pp_trace_kernel (the SHIPPED kernel body, run under the
// CPU shim) vs genomsa::align_profiles (the CPU reference that fixes the
// semantics). Asserts score, aligned span and column CIGAR all agree -- the
// kernel records decisions in direction bytes, the reference re-derives them
// by float comparison, so agreement is meaningful.
//
// Build:  g++ -O2 -std=c++17 -DGENOALIGNER_HIP_SHIM \
//           -I tests/parity/hip_cpu_shim -I include \
//           tests/parity/msa_pp_parity.cpp src/msa/msa_ref.cpp
#include <genoaligner/msa/msa.hpp>
#include <cstdio>
#include <cstring>
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

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); ++fails; } } while (0)

// genomsa::Profile -> SoA device view (host memory under the shim).
struct SoAHolder {
    std::vector<float> c[5];
    std::vector<float> occ;
    MsaProfileView view;
};
static SoAHolder to_soa(const Profile& p) {
    SoAHolder h;
    h.occ = p.occ;
    for (int b = 0; b < 5; ++b) {
        h.c[b].resize(p.cols.size());
        for (size_t i = 0; i < p.cols.size(); ++i) h.c[b][i] = p.cols[i][b];
    }
    for (int b = 0; b < 5; ++b) h.view.cols[b] = h.c[b].data();
    h.view.occ = h.occ.data();
    h.view.len = p.ncols();
    return h;
}

static MsaPPParams to_params(const Params& P) {
    return {P.match, P.ts, P.tv, P.gap_open, P.gap_extend,
            P.free_end_gaps ? 1 : 0};
}

// Run the shipped kernel body on one pair, exactly as a one-thread-per-pair
// launch would execute it.
static bool run_kernel(const Profile& A, const Profile& B, const Params& P,
                       MsaPPResult& res, std::string& cigar) {
    SoAHolder ha = to_soa(A), hb = to_soa(B);
    MsaPPPair pp{ha.view, hb.view};

    const size_t dir_sz = (size_t)(A.ncols() + 1) * (B.ncols() + 1);
    const size_t scr_sz = (size_t)3 * (B.ncols() + 1) + 2 * (A.ncols() + 1);
    const int    cap    = A.ncols() + B.ncols() + 4;
    std::vector<uint8_t> dirs(dir_sz), cig(cap);
    std::vector<float>   scr(scr_sz);
    int meta[2];
    const size_t zero[1] = {0}, av_d[1] = {dir_sz}, av_s[1] = {scr_sz};

    blockIdx = uint3{0,0,0}; threadIdx = uint3{0,0,0};
    blockDim = dim3{1,1,1};  gridDim  = dim3{1,1,1};
    msa_pp_trace_kernel(&pp, to_params(P), &res, dirs.data(), zero, av_d,
                        cig.data(), zero, meta, scr.data(), zero, av_s, 1, cap);
    static const char ops[] = "MID";
    for (int k = meta[1] - 1; k >= 0; --k) cigar.push_back(ops[cig[k]]);
    return res.score != MSA_PP_TOO_BIG;
}

// Same pair through the wavefront kernel: block-per-pair, so the shim's
// run_block spawns 256 real host threads with a real block barrier -- the
// anti-diagonal synchronisation is genuinely exercised, not paraphrased.
static bool run_kernel_wf(const Profile& A, const Profile& B, const Params& P,
                          MsaPPResult& res, std::string& cigar) {
    SoAHolder ha = to_soa(A), hb = to_soa(B);
    MsaPPPair pp{ha.view, hb.view};

    const size_t dir_sz = (size_t)(A.ncols() + 1) * (B.ncols() + 1);
    const size_t scr_sz = (size_t)11 * (A.ncols() + 1) + 2 * (B.ncols() + 1);
    const int    cap    = A.ncols() + B.ncols() + 4;
    std::vector<uint8_t> dirs(dir_sz), cig(cap);
    std::vector<float>   scr(scr_sz);
    int meta[2];
    const size_t zero[1] = {0}, av_d[1] = {dir_sz}, av_s[1] = {scr_sz};

    blockIdx = uint3{0,0,0}; threadIdx = uint3{0,0,0};
    blockDim = dim3{256,1,1};  gridDim = dim3{1,1,1};
    shim::run_block(256, [&] {
        msa_pp_trace_kernel_wf(&pp, to_params(P), &res, dirs.data(), zero, av_d,
                               cig.data(), zero, meta, scr.data(), zero, av_s,
                               1, cap);
    });
    static const char ops[] = "MID";
    for (int k = meta[1] - 1; k >= 0; --k) cigar.push_back(ops[cig[k]]);
    return res.score != MSA_PP_TOO_BIG;
}

static void check_pair(const Profile& A, const Profile& B, const Params& P,
                       const char* tag) {
    genomsa::AlignResult ref = genomsa::align_profiles(A, B, P);
    MsaPPResult k; std::string kcig;
    bool ok = run_kernel(A, B, P, k, kcig);
    CHECK(ok, "%s: kernel too_big", tag);
    CHECK(k.score == ref.score, "%s: score kernel=%g ref=%g", tag, k.score, ref.score);
    CHECK(k.ai == ref.ai && k.aj == ref.aj && k.bi == ref.bi && k.bj == ref.bj,
          "%s: span k=(%d,%d)-(%d,%d) ref=(%d,%d)-(%d,%d)",
          tag, k.ai, k.aj, k.bi, k.bj, ref.ai, ref.aj, ref.bi, ref.bj);
    CHECK(kcig == ref.cigar, "%s: cigar k=%s ref=%s", tag, kcig.c_str(), ref.cigar.c_str());
    MsaPPResult w; std::string wcig;
    bool wok = run_kernel_wf(A, B, P, w, wcig);
    CHECK(wok, "%s: wf kernel too_big", tag);
    CHECK(w.score == ref.score, "%s: wf score=%g ref=%g", tag, w.score, ref.score);
    CHECK(w.ai == ref.ai && w.aj == ref.aj && w.bi == ref.bi && w.bj == ref.bj,
          "%s: wf span k=(%d,%d)-(%d,%d) ref=(%d,%d)-(%d,%d)",
          tag, w.ai, w.aj, w.bi, w.bj, ref.ai, ref.aj, ref.bi, ref.bj);
    CHECK(wcig == ref.cigar, "%s: wf cigar k=%s ref=%s", tag, wcig.c_str(), ref.cigar.c_str());
}

static std::mt19937 rng(4242);
static std::string rand_dna(int len) {
    static const char b[] = "ACGT";
    std::string s; for (int i = 0; i < len; ++i) s += b[rng() % 4]; return s;
}
// a related sequence: substitutions + small indels
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

int main() {
    Params P;

    // ---- single-sequence profiles (pairwise semiglobal) ----
    check_pair(genomsa::profile_from_seq("ACGTACGT"),
               genomsa::profile_from_seq("ACGTACGT"), P, "identical");
    check_pair(genomsa::profile_from_seq("ACGTACGT"),
               genomsa::profile_from_seq("ACGTTTTACGT"), P, "insertion");
    check_pair(genomsa::profile_from_seq("ACGTACGTACGTACGTACGT"),
               genomsa::profile_from_seq("GTACGTAC"), P, "fragment");
    check_pair(genomsa::profile_from_seq("ANRYSWKB"),
               genomsa::profile_from_seq("ACGTACGT"), P, "iupac");

    // ---- merged profiles (fractional counts + gap columns) ----
    {
        Profile a1 = genomsa::profile_from_seq("ACGTACGT");
        Profile a2 = genomsa::profile_from_seq("ACGTTCGT");
        auto r = genomsa::align_profiles(a1, a2, P);
        Profile ab = genomsa::merge_profiles(a1, a2, r);
        Profile c  = genomsa::profile_from_seq("ACGTACGTAA");
        check_pair(ab, c, P, "profile-vs-seq");
        Profile d  = genomsa::profile_from_seq("TTGCGCGC");
        check_pair(ab, d, P, "profile-vs-seq2");
    }

    // ---- global mode ----
    {
        Params G = P; G.free_end_gaps = false;
        check_pair(genomsa::profile_from_seq("ACGTACGT"),
                   genomsa::profile_from_seq("ACGTTTACGT"), G, "global-indel");
        check_pair(genomsa::profile_from_seq("ACGTACGT"),
                   genomsa::profile_from_seq("TTTT"), G, "global-divergent");
    }

    // ---- random sweep: profiles from small merges ----
    for (int t = 0; t < 150; ++t) {
        std::string anc = rand_dna(20 + rng() % 60);
        Profile a1 = genomsa::profile_from_seq(mutate(anc, 10, 5, 5));
        Profile a2 = genomsa::profile_from_seq(mutate(anc, 10, 5, 5));
        auto ra = genomsa::align_profiles(a1, a2, P);
        Profile A = genomsa::merge_profiles(a1, a2, ra);
        std::string anc2 = rand_dna(20 + rng() % 60);
        Profile b1 = genomsa::profile_from_seq(mutate(anc2, 10, 5, 5));
        Profile b2 = genomsa::profile_from_seq(mutate(anc2, 10, 5, 5));
        auto rb = genomsa::align_profiles(b1, b2, P);
        Profile Bp = genomsa::merge_profiles(b1, b2, rb);
        char tag[32]; snprintf(tag, sizeof tag, "rand-%d", t);
        check_pair(A, Bp, P, tag);
    }
    // random single-seq sweep too
    for (int t = 0; t < 150; ++t) {
        char tag[32]; snprintf(tag, sizeof tag, "randseq-%d", t);
        check_pair(genomsa::profile_from_seq(rand_dna(5 + rng() % 60)),
                   genomsa::profile_from_seq(rand_dna(5 + rng() % 60)), P, tag);
    }

    if (!fails) printf("RESULT: PASS -- kernel == CPU reference on all cases\n");
    else        printf("%d FAILURES\n", fails);
    return fails;
}
