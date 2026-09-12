// genoaligner — Smith-Waterman on GPU: the warp-per-row kernel.
//
// What this run establishes, stated narrowly:
//   * correctness of the shipped kernel at blockDim > 1 -- one warp per pair,
//     columns tiled across lanes, the intra-row dependency carried by the scan
//     described in include/genoaligner/backend/sw_kernel.hip;
//   * with --bench, the determinism of its timing: 7 repetitions of IDENTICAL
//     work, reported as a spread, because the WFA kernel failed exactly this
//     check (2.5x between identical launches; docs/RESULTADO_B6_TCUPS.md).
//
// The CPU parity gate (tests/sw/test_sw_parity.cpp) already ran this same body
// under real threaded emulation at warpSize 32 AND 64, so this job is the check
// that nothing is lost in translation to a real device -- which is where both SW
// GPU faults found so far lived.

#include "genoaligner/backend/sw_kernel.hip"
// The kernel BODY lives in the _impl header. hipLaunchKernelGGL needs both the
// declaration and the definition visible in the translation unit that launches it, or
// the device stub is undefined at link time (which is how this line came to be needed).
#include "genoaligner/backend/sw_kernel_impl.hip"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include <random>

using genoaligner::SWParams;
using genoaligner::SWPairView;
using genoaligner::SWResult;
using genoaligner::sw_score_kernel;

static int g_fail = 0;

// Independent full-matrix reference (same as tests/sw/test_sw_reference.cpp).
static int ref_sw(const std::string& text, const std::string& pattern,
                  SWParams p, int* oi, int* oj)
{
    const int m = (int)text.size(), n = (int)pattern.size();
    if (m == 0 || n == 0) { if (oi) *oi = -1; if (oj) *oj = -1; return 0; }
    std::vector<int> H((size_t)(m+1)*(size_t)(n+1),0), E(H.size(),0), F(H.size(),0);
    auto at = [&](std::vector<int>& M, int i, int j)->int& {
        return M[(size_t)i*(size_t)(n+1)+(size_t)j]; };
    int best = 0, bi = -1, bj = -1;
    for (int i = 1; i <= m; ++i)
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
    if (oi) *oi = bi; if (oj) *oj = bj;
    return best;
}

struct Case { std::string text, pattern, label; };

int main(int argc, char** argv)
{
    const bool bench = (argc > 1 && std::strcmp(argv[1], "--bench") == 0);
    printf("genoaligner — Smith-Waterman on GPU (warp-per-row)%s\n",
           bench ? " [bench]" : "");

    int nsm = 0;
    hipDeviceProp_t prop{};
    if (hipGetDeviceCount(&nsm) != hipSuccess || nsm == 0 ||
        hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        printf("  SKIP: no device visible; nothing verified here.\n");
        printf("RESULT: SKIP -- needs a GPU (exit 77)\n");
        return 77;
    }
    int dev_warp = 0;
    hipDeviceGetAttribute(&dev_warp, hipDeviceAttributeWarpSize, 0);
    printf("  device: %s   warpSize=%d\n\n", prop.name, dev_warp);

    const SWParams P{1, -1, 2, 1};

    std::vector<Case> cases = {
        {"A", "A", "single base"},
        {"ACGTACGT", "ACGTACGT", "8 identical"},
        {"ACGTACGT", "TGCATGCA", "shares one A"},
        {"AAAAAAAA", "CCCCCCCC", "disjoint"},
        {"ACGTTTACGT", "ACGTACGT", "gap of 2 inside a match"},
        {std::string(10,'T') + "ACGTACGTACGT" + std::string(10,'G'),
         std::string(10,'C') + "ACGTACGTACGT" + std::string(20,'A'),
         "core with divergent flanks"},
        {"ACACACAC", "ACAC", "tandem repeat"},
        {std::string(33,'A'), std::string(33,'A'), "n=33 (crosses a 32-lane tile)"},
        {std::string(65,'A'), std::string(65,'C'), "n=65 (crosses a 64-lane tile)"},
    };
    // A longer case, to exercise the tiled row walk well past one warp tile.
    {
        std::string t, q;
        for (int i = 0; i < 400; ++i) { t.push_back("ACGT"[i % 4]); }
        q = std::string(20, 'T') + t.substr(100, 120) + std::string(20, 'G');
        cases.push_back({t, q, "400 x 160 embedded core"});
    }
    // Randomised batch (fixed seed): mixed sizes, some past two warp tiles.
    {
        std::mt19937 rng(20260912u);
        const char* alpha = "ACGT";
        for (int k = 0; k < 200; ++k) {
            const int m = 1 + (int)(rng() % 200), n = 1 + (int)(rng() % 200);
            std::string t, q;
            for (int i = 0; i < m; ++i) t.push_back(alpha[rng() % 4]);
            for (int i = 0; i < n; ++i) q.push_back(alpha[rng() % 4]);
            cases.push_back({t, q, "random"});
        }
    }

    const int N = (int)cases.size();
    std::vector<SWPairView> hv((size_t)N);
    std::vector<SWResult>   hr((size_t)N, SWResult{-1,-1,-1});

    // POINTERS, NOT COPIES: SWPairView holds const char*. The struct must be built from
    // DEVICE pointers, so the bytes go to the device first. This is the bug that
    // faulted the first GPU run (see tests/sw/test_sw_dbg.cpp) and it is the same
    // mistake the project already documented for WFA's PairView.
    std::vector<char*> d_txt((size_t)N, nullptr), d_pat((size_t)N, nullptr);
    for (int i = 0; i < N; ++i) {
        const std::string& t = cases[(size_t)i].text;
        const std::string& q = cases[(size_t)i].pattern;
        if (hipMalloc(&d_txt[(size_t)i], t.size() ? t.size() : 1) != hipSuccess ||
            hipMalloc(&d_pat[(size_t)i], q.size() ? q.size() : 1) != hipSuccess) {
            printf("  hipMalloc(chars) failed\n"); return 1;
        }
        hipMemcpy(d_txt[(size_t)i], t.data(), t.size(), hipMemcpyHostToDevice);
        hipMemcpy(d_pat[(size_t)i], q.data(), q.size(), hipMemcpyHostToDevice);
        hv[(size_t)i].text = d_txt[(size_t)i]; hv[(size_t)i].text_len = (int)t.size();
        hv[(size_t)i].pattern = d_pat[(size_t)i]; hv[(size_t)i].pattern_len = (int)q.size();
    }

    SWPairView* d_pairs = nullptr;
    SWResult*   d_res   = nullptr;
    if (hipMalloc(&d_pairs, (size_t)N * sizeof(SWPairView)) != hipSuccess ||
        hipMalloc(&d_res,   (size_t)N * sizeof(SWResult))   != hipSuccess) {
        printf("  hipMalloc failed\n"); return 1;
    }
    hipMemcpy(d_pairs, hv.data(), (size_t)N * sizeof(SWPairView), hipMemcpyHostToDevice);
    hipMemset(d_res, 0, (size_t)N * sizeof(SWResult));

    // Launch contract (one place: sw_kernel.hip). Shared memory is sized by the
    // LARGEST pattern in the batch, so one launch config covers all.
    int max_n = 0;
    for (const auto& c : cases) if ((int)c.pattern.size() > max_n) max_n = (int)c.pattern.size();
    const int ints_per_w = genoaligner::sw_smem_ints_per_warp(max_n);
    const int block      = genoaligner::SW_WARPS_PER_BLOCK * dev_warp;
    const int grid       = (N + genoaligner::SW_WARPS_PER_BLOCK - 1)
                           / genoaligner::SW_WARPS_PER_BLOCK;
    const size_t smem    = (size_t)genoaligner::SW_WARPS_PER_BLOCK
                           * (size_t)ints_per_w * sizeof(int);
    printf("  launch: grid=%d block=%d (%d warps x %d lanes)  smem=%zu B\n\n",
           grid, block, genoaligner::SW_WARPS_PER_BLOCK, dev_warp, smem);

    auto launch = [&] {
        hipLaunchKernelGGL(sw_score_kernel, dim3((unsigned)grid), dim3((unsigned)block),
                           smem, (hipStream_t)0, d_pairs, P, d_res, N, ints_per_w);
    };

    launch();
    hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
        printf("  LAUNCH FAILED: %s (smem=%zu B, max_n=%d)\n", hipGetErrorString(le), smem, max_n);
        return 1;
    }
    hipDeviceSynchronize();

    hipMemcpy(hr.data(), d_res, (size_t)N * sizeof(SWResult), hipMemcpyDeviceToHost);

    printf("-- kernel vs independent reference --\n");
    int shown = 0;
    for (int i = 0; i < N; ++i) {
        int ri, rj;
        const int ref = ref_sw(cases[(size_t)i].text, cases[(size_t)i].pattern, P, &ri, &rj);
        const SWResult& r = hr[(size_t)i];
        const bool ok = (r.score == ref);
        if (!ok || shown < 12) {
            printf("  %-38s ref=%4d kernel=%4d  %s\n",
                   cases[(size_t)i].label.c_str(), ref, r.score, ok ? "ok" : "MISMATCH");
            ++shown;
        }
        if (!ok) ++g_fail;
    }
    printf("  (%d/%d agree)\n", N - g_fail, N);

    if (bench && g_fail == 0) {
        // Determinism: the standard B6 set for this kernel's failure mode. Seven
        // reps of IDENTICAL work; the spread is the claim, so it is printed as a
        // spread rather than hidden inside a mean.
        printf("\n-- determinism: 7 identical launches --\n");
        double mn = 1e30, mx = 0, sum = 0;
        for (int rep = 0; rep < 7; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            launch();
            hipDeviceSynchronize();
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count();
            printf("  rep %d: %.3f ms\n", rep, ms);
            if (ms < mn) mn = ms;
            if (ms > mx) mx = ms;
            sum += ms;
        }
        const double spread = (mn > 0) ? (mx - mn) / mn : 0.0;
        printf("  min=%.3f ms  max=%.3f ms  mean=%.3f ms  spread=%.1f%%\n",
               mn, mx, sum / 7, 100.0 * spread);
        printf("  criterion (B6): spread < 5%%  ->  %s\n",
               spread < 0.05 ? "PASS" : "FAIL");
        if (spread >= 0.05) ++g_fail;
    }

    for (int i = 0; i < N; ++i) { hipFree(d_txt[(size_t)i]); hipFree(d_pat[(size_t)i]); }
    hipFree(d_pairs); hipFree(d_res);
    printf("\n");
    if (g_fail == 0) {
        printf("RESULT: PASS -- SW warp-per-row kernel matches the reference on GPU\n");
        return 0;
    }
    printf("RESULT: FAIL (%d)\n", g_fail);
    return 1;
}
