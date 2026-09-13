// genoaligner — Smith-Waterman kernel parity gate (warp-per-row).
//
// WHAT IT COMPARES
// ----------------
// The SHIPPED kernel body executed through the CPU shim's warp emulation
// (shim::run_block: one host thread per lane, real exchanges for __shfl_*_sync)
// against the independent full-matrix reference below. Because the emulation is
// real concurrency, this gate covers blockDim > 1 -- the configuration the old
// row-walk kernel could never be checked in.
//
// The gate runs EVERY case at BOTH emulated warp widths (32 and 64): the shipped
// kernel must be correct on NVIDIA and AMD warps alike, and a warp-width
// assumption caught here never reaches a GPU.
//
// NEGATIVE CONTROL, BUILT IN
// --------------------------
// The plan's criterion: parity must BREAK if the synchronisation is removed.
// shim::g_shuffle_passthrough makes every __shfl_*_sync return the caller's own
// value -- i.e. the cross-lane exchange silently vanishes. The last section runs
// the full case set with the exchange disabled and REQUIRES at least one
// mismatch. If the gate stayed green with the dataflow cut, it was never
// measuring the dataflow; a clean pass there is reported as a gate defect, not
// as good news.

#include "genoaligner/backend/sw_kernel.hip"
#include "genoaligner/backend/sw_kernel_impl.hip"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <random>

// The shim declares the launch-config globals extern; this file is the launch
// owner, so it defines them (threadIdx is per-lane: thread_local).
#ifdef GENOALIGNER_HIP_SHIM
thread_local uint3 threadIdx{0, 0, 0};
uint3 blockIdx{0, 0, 0};
dim3  blockDim{1, 1, 1};
dim3  gridDim{1, 1, 1};
#endif

// The reference, duplicated here minimally so this gate has no build dependency on the
// test binary. Kept identical to tests/sw/test_sw_reference.cpp -- if they ever diverge,
// the divergence is the bug.
static int ref_sw(const std::string& text, const std::string& pattern,
                  genoaligner::SWParams p, int* oi, int* oj)
{
    const int m = (int)text.size(), n = (int)pattern.size();
    if (m == 0 || n == 0) { if (oi) *oi = -1; if (oj) *oj = -1; return 0; }
    std::vector<int> H((size_t)(m+1)*(size_t)(n+1),0), E(H.size(),0), F(H.size(),0);
    auto at = [&](std::vector<int>& M, int i, int j)->int& { return M[(size_t)i*(size_t)(n+1)+(size_t)j]; };
    int best = 0, bi = -1, bj = -1;
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            const int s = (text[(size_t)i-1] == pattern[(size_t)j-1]) ? p.match : p.mismatch;
            int e = at(H,i,j-1) - p.gap_open; { int t = at(E,i,j-1) - p.gap_extend; if (t > e) e = t; }
            int f = at(H,i-1,j) - p.gap_open; { int t = at(F,i-1,j) - p.gap_extend; if (t > f) f = t; }
            int h = at(H,i-1,j-1) + s;
            if (e > h) h = e;
            if (f > h) h = f;
            if (h < 0) h = 0;
            at(H,i,j)=h; at(E,i,j)=e; at(F,i,j)=f;
            if (h > best) { best = h; bi = i-1; bj = j-1; }
        }
    }
    if (oi) *oi = bi; if (oj) *oj = bj;
    return best;
}

struct Case { std::string text, pattern, label; };

// Run the whole case list through ONE emulated grid: pairs are packed
// SW_WARPS_PER_BLOCK per emulated block, so the run also checks that adjacent
// warps in a block do not leak into each other's shared slices. Returns the
// kernel results for every case.
static void run_grid(const std::vector<Case>& cases, genoaligner::SWParams prm,
                     int wsize, std::vector<genoaligner::SWResult>& out)
{
    const int npairs = (int)cases.size();
    std::vector<genoaligner::SWPairView> pv((size_t)npairs);
    int max_n = 0;
    for (int i = 0; i < npairs; ++i) {
        pv[(size_t)i].text        = cases[(size_t)i].text.data();
        pv[(size_t)i].text_len    = (int)cases[(size_t)i].text.size();
        pv[(size_t)i].pattern     = cases[(size_t)i].pattern.data();
        pv[(size_t)i].pattern_len = (int)cases[(size_t)i].pattern.size();
        if (pv[(size_t)i].pattern_len > max_n) max_n = pv[(size_t)i].pattern_len;
    }
    out.assign((size_t)npairs, genoaligner::SWResult{-1, -1, -1});

    const int nthreads   = genoaligner::SW_WARPS_PER_BLOCK * wsize;
    const int ints_per_w = genoaligner::sw_smem_ints_per_warp(max_n);

    // Shared memory, sized exactly as a real launch would size it: undersizing
    // must show up as a fault under ASan, not hide inside a bigger buffer.
    shim::smem_vec().assign((size_t)genoaligner::SW_WARPS_PER_BLOCK * (size_t)ints_per_w, 0);

    warpSize = wsize;
    blockDim = dim3{(unsigned)nthreads, 1, 1};

    const int nblocks = (npairs + genoaligner::SW_WARPS_PER_BLOCK - 1)
                        / genoaligner::SW_WARPS_PER_BLOCK;
    for (int b = 0; b < nblocks; ++b) {
        blockIdx = uint3{(unsigned)b, 0, 0};
        shim::run_block(nthreads, [&] {
            genoaligner::sw_score_kernel(pv.data(), prm, out.data(), npairs, ints_per_w);
        });
    }
}

int main()
{
    printf("genoaligner — Smith-Waterman parity gate (warp-per-row, CPU emulation)\n\n");
    const genoaligner::SWParams SWP{1, -1, 2, 1};

    std::vector<Case> cases = {
        {"A", "A", "single base"},
        {"ACGTACGT", "ACGTACGT", "8 identical"},
        {"ACGTACGT", "TGCATGCA", "shares one A"},
        {"AAAAAAAA", "CCCCCCCC", "disjoint"},
        {"ACGTTTACGT", "ACGTACGT", "gap of 2 inside a match"},
        {std::string(10,'T') + "ACGTACGTACGT" + std::string(10,'G'),
         std::string(10,'C') + "ACGTACGTACGT" + std::string(10,'A'),
         "core with divergent flanks"},
        {"ACACACAC", "ACAC", "tandem repeat"},
        {"", "", "both empty"},
        {"ACGT", "", "empty pattern"},
        {"", "ACGT", "empty text"},
        // Multi-tile cases: pattern longer than one AND two warp widths, so the
        // tile-carry path (lane 0 diagonal, U carry) is exercised at both 32 and 64.
        {std::string(33,'A'), std::string(33,'A'), "n=33: crosses a 32-lane tile"},
        {std::string(65,'A'), std::string(65,'C'), "n=65: crosses a 64-lane tile"},
        {std::string(200,'G') + "ACGTACGT" + std::string(50,'T'),
         std::string(40,'A') + "ACGTACGT" + std::string(90,'C'), "n=170,m=258 core"},
    };
    {
        std::string t, q;
        for (int i = 0; i < 400; ++i) t.push_back("ACGT"[i % 4]);
        q = std::string(20,'T') + t.substr(100, 120) + std::string(20,'G');
        cases.push_back({t, q, "400 x 160 embedded core"});
    }
    // Randomised batch: fixed seed, mixed sizes 0..130 so every emulated block
    // mixes short and multi-tile pairs.
    {
        std::mt19937 rng(20260912u);
        const char* alpha = "ACGT";
        for (int k = 0; k < 96; ++k) {
            const int m = (int)(rng() % 131), n = (int)(rng() % 131);
            std::string t, q;
            for (int i = 0; i < m; ++i) t.push_back(alpha[rng() % 4]);
            for (int i = 0; i < n; ++i) q.push_back(alpha[rng() % 4]);
            cases.push_back({t, q, "random"});
        }
    }

    // Reference results (score AND end coordinate).
    const int nc = (int)cases.size();
    std::vector<int> ref_s((size_t)nc), ref_i((size_t)nc), ref_j((size_t)nc);
    for (int i = 0; i < nc; ++i)
        ref_s[(size_t)i] = ref_sw(cases[(size_t)i].text, cases[(size_t)i].pattern,
                                  SWP, &ref_i[(size_t)i], &ref_j[(size_t)i]);

    int total_fail = 0;
    for (int wsize : {32, 64}) {
        printf("-- emulated warpSize = %d, blockDim = %d (%d warps/block) --\n",
               wsize, genoaligner::SW_WARPS_PER_BLOCK * wsize,
               genoaligner::SW_WARPS_PER_BLOCK);
        std::vector<genoaligner::SWResult> res;
        run_grid(cases, SWP, wsize, res);

        int fails = 0;
        for (int i = 0; i < nc; ++i) {
            const genoaligner::SWResult& r = res[(size_t)i];
            bool ok = (r.score == ref_s[(size_t)i]);
            // Coordinates: the score is the claim under test, but a reported
            // endpoint must at least be in range, and a zero score must say so.
            if (ok) {
                if (r.score == 0) ok = (r.end_i == -1 && r.end_j == -1);
                else ok = (r.end_i >= 0 && r.end_i < (int)cases[(size_t)i].text.size() &&
                           r.end_j >= 0 && r.end_j < (int)cases[(size_t)i].pattern.size());
            }
            if (!ok) {
                ++fails;
                if (fails <= 8)
                    printf("  MISMATCH [%s] ref=%d k=%d end=(%d,%d) m=%zu n=%zu\n",
                           cases[(size_t)i].label.c_str(), ref_s[(size_t)i], r.score,
                           r.end_i, r.end_j, cases[(size_t)i].text.size(),
                           cases[(size_t)i].pattern.size());
            }
        }
        printf("  %d/%d cases agree with the reference at warpSize=%d\n\n",
               nc - fails, nc, wsize);
        total_fail += fails;
    }

    // NEGATIVE CONTROL: kill the cross-lane exchange and require the gate to go
    // red. If every case still passes, the gate is not measuring what it claims.
    printf("-- negative control: shuffles return own value (exchange removed) --\n");
    shim::g_shuffle_passthrough = true;
    std::vector<genoaligner::SWResult> broken;
    run_grid(cases, SWP, 32, broken);
    shim::g_shuffle_passthrough = false;
    int broken_wrong = 0;
    for (int i = 0; i < nc; ++i)
        if (broken[(size_t)i].score != ref_s[(size_t)i]) ++broken_wrong;
    printf("  %d/%d cases went wrong with the exchange disabled\n", broken_wrong, nc);
    const bool control_ok = (broken_wrong > 0);
    if (!control_ok)
        printf("  !!! every case still passed -- this gate was not measuring the\n"
               "      cross-lane dataflow at all. Treat as a gate DEFECT.\n");

    printf("\n");
    if (total_fail == 0 && control_ok) {
        printf("RESULT: PASS -- kernel matches the reference at warpSize 32 AND 64,\n");
        printf("        blockDim>1, and the gate provably fails without the exchange\n");
        return 0;
    }
    printf("RESULT: FAIL (mismatches=%d, negative-control %s)\n",
           total_fail, control_ok ? "ok" : "DEFECT");
    return 1;
}
