// genoaligner — SW throughput: our GPU score kernel vs parasail (CPU, SIMD).
//
// METHODOLOGY IS B6's, ON PURPOSE (docs/RESULTADO_B6_TCUPS.md):
//   * everything inside ONE job -- no numbers are ever compared across jobs;
//   * 7 reps; ALL THREE columns measured inside every rep (interleaved), so
//     intra-job drift hits each column equally;
//   * GPU time via hipEvent (device execution) plus wall for context -- this
//     cluster time-slices GPUs across jobs (shard gres, see
//     docs/RESULTADO_H8_SW1_WARP.md §3);
//   * reported as mean +/- sd, min, max, spread -- never a bare ratio;
//   * and every rep CHECKS the scores: a fast wrong answer fails, not passes.
//
// WHAT IS BEING COMPARED -- stated plainly, because this is the number that
// will get quoted:
//   * "gpu":   our sw_score_kernel, whole batch in one launch, device time.
//   * "p1":    parasail_sw (dispatched to the best ISA this CPU supports),
//              ONE host thread, pair after pair.
//   * "p8":    the same parasail calls split across 8 host threads -- the
//              ceiling a real CPU pipeline would see, not a library feature.
// This is GPU-vs-CPU, not an apples claim about algorithms: parasail is the
// standard vectorised Smith-Waterman and the honest CPU baseline.
//
// PARASAIL GAP CONVENTION, VERIFIED: parasail_sw's (open, gap) uses
// open + (L-1)*gap for a gap of length L -- the SAME as our SWScoring
// (pin: ACGTTTACGT vs ACGTACGT under {1,-1,2,1} gives 5, not 4). So the
// scoring arguments pass through unmapped.

#include "genoaligner/backend/sw_kernel.hip"
#include "genoaligner/backend/sw_kernel_impl.hip"

#include <parasail.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

using genoaligner::SWParams;
using genoaligner::SWPairView;
using genoaligner::SWResult;
using genoaligner::sw_score_kernel;

struct Pair { std::string text, pattern; };

// The same independent serial reference the SW gates use -- the bench must not
// trust either competitor's score.
static int ref_sw(const std::string& t, const std::string& p, const SWParams& s)
{
    const int m = (int)t.size(), n = (int)p.size();
    if (m <= 0 || n <= 0) return 0;
    std::vector<int> H((size_t)(n + 1), 0), E((size_t)(n + 1), 0);
    int best = 0;
    for (int i = 1; i <= m; ++i) {
        int prevDiag = 0;    // H[i-1][j-1]
        int hPrev = 0;       // H[i][j-1]
        int F = 0;           // across-gap: F[i][j] = max(F[i][j-1]-ge, H[i][j-1]-go)
        for (int j = 1; j <= n; ++j) {
            const int Hij = H[(size_t)j];    // H[i-1][j]
            E[(size_t)j] = std::max(E[(size_t)j] - s.gap_extend, Hij - s.gap_open);
            F = std::max(F - s.gap_extend, hPrev - s.gap_open);
            const int sub = prevDiag + (t[(size_t)(i - 1)] == p[(size_t)(j - 1)]
                                        ? s.match : s.mismatch);
            const int h = std::max(0, std::max(sub, std::max(E[(size_t)j], F)));
            prevDiag = Hij;
            H[(size_t)j] = h;
            hPrev = h;
            if (h > best) best = h;
        }
    }
    return best;
}

struct Stats { double mean, sd, mn, mx; };
static Stats stats(const std::vector<double>& v)
{
    Stats s{0, 0, 1e300, 0};
    for (double x : v) { s.mean += x; s.mn = std::min(s.mn, x); s.mx = std::max(s.mx, x); }
    s.mean /= (double)v.size();
    for (double x : v) s.sd += (x - s.mean) * (x - s.mean);
    s.sd = std::sqrt(s.sd / (double)v.size());
    return s;
}

int main(int argc, char** argv)
{
    const int reps = (argc > 1) ? std::atoi(argv[1]) : 7;

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        printf("  SKIP: no device visible; nothing verified here.\n");
        return 77;
    }
    const int dev_warp = prop.warpSize;
    printf("genoaligner — SW throughput: GPU score kernel vs parasail\n");
    printf("  device: %s   warpSize=%d\n\n", prop.name, dev_warp);

    // ---- workload -----------------------------------------------------------
    // Three size classes, mixed, fixed seed. Small enough that parasail-1thread
    // finishes a rep in seconds; big enough that a GPU launch is not dominated
    // by its tail. GCUPS below counts SUM(m*n) -- the cells actually visited.
    const SWParams P{1, -1, 2, 1};   // the scheme the gates and oracle pin
    std::vector<Pair> cases;
    {
        std::mt19937 rng(20260912u);
        const char* alpha = "ACGT";
        auto gen = [&](int m, int n, int extra) {
            for (int k = 0; k < extra; ++k) {
                std::string t, q;
                for (int i = 0; i < m; ++i) t.push_back(alpha[rng() % 4]);
                for (int i = 0; i < n; ++i) q.push_back(alpha[rng() % 4]);
                cases.push_back({t, q});
            }
        };
        gen(200, 200, 400);    // bulk
        gen(512, 512, 150);    // mid
        gen(1000, 1000, 50);   // tail
    }
    const int N = (int)cases.size();
    double cells = 0;
    for (const auto& c : cases)
        cells += (double)c.text.size() * (double)c.pattern.size();
    printf("  workload: %d pairs, %.1f Mcells/rep\n", N, cells / 1e6);

    // ---- ground truth: compute the reference scores ONCE --------------------
    std::vector<int> ref((size_t)N);
    for (int i = 0; i < N; ++i)
        ref[(size_t)i] = ref_sw(cases[(size_t)i].text, cases[(size_t)i].pattern, P);

    // ---- parasail side ------------------------------------------------------
    parasail_matrix_t* mat = parasail_matrix_create("ACGT", P.match, P.mismatch);
    std::vector<int> ps((size_t)N);

    auto parasail_range = [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i) {
            parasail_result_t* r = parasail_sw(
                cases[(size_t)i].text.c_str(), (int)cases[(size_t)i].text.size(),
                cases[(size_t)i].pattern.c_str(), (int)cases[(size_t)i].pattern.size(),
                P.gap_open, P.gap_extend, mat);
            ps[(size_t)i] = parasail_result_get_score(r);
            parasail_result_free(r);
        }
    };
    auto parasail_all = [&](int nthreads) {
        if (nthreads <= 1) { parasail_range(0, N); return; }
        std::vector<std::thread> ts;
        const int chunk = (N + nthreads - 1) / nthreads;
        for (int t = 0; t < nthreads; ++t) {
            const int lo = t * chunk, hi = std::min(N, lo + chunk);
            if (lo < hi) ts.emplace_back(parasail_range, lo, hi);
        }
        for (auto& t : ts) t.join();
    };

    // ---- GPU side -----------------------------------------------------------
    std::vector<SWPairView> hv((size_t)N);
    std::vector<SWResult>   hr((size_t)N, SWResult{-1, -1, -1});
    std::vector<char*> d_txt((size_t)N), d_pat((size_t)N);
    for (int i = 0; i < N; ++i) {
        const Pair& c = cases[(size_t)i];
        if (hipMalloc(&d_txt[(size_t)i], c.text.size() ? c.text.size() : 1) != hipSuccess ||
            hipMalloc(&d_pat[(size_t)i], c.pattern.size() ? c.pattern.size() : 1) != hipSuccess) {
            printf("  hipMalloc(chars) failed\n"); return 1;
        }
        hipMemcpy(d_txt[(size_t)i], c.text.data(), c.text.size(), hipMemcpyHostToDevice);
        hipMemcpy(d_pat[(size_t)i], c.pattern.data(), c.pattern.size(), hipMemcpyHostToDevice);
        hv[(size_t)i].text = d_txt[(size_t)i]; hv[(size_t)i].text_len = (int)c.text.size();
        hv[(size_t)i].pattern = d_pat[(size_t)i]; hv[(size_t)i].pattern_len = (int)c.pattern.size();
    }
    SWPairView* d_pairs = nullptr;
    SWResult*   d_res   = nullptr;
    if (hipMalloc(&d_pairs, (size_t)N * sizeof(SWPairView)) != hipSuccess ||
        hipMalloc(&d_res,   (size_t)N * sizeof(SWResult))   != hipSuccess) {
        printf("  hipMalloc failed\n"); return 1;
    }
    hipMemcpy(d_pairs, hv.data(), (size_t)N * sizeof(SWPairView), hipMemcpyHostToDevice);
    hipMemset(d_res, 0, (size_t)N * sizeof(SWResult));

    int max_n = 0;
    for (const auto& c : cases) max_n = std::max(max_n, (int)c.pattern.size());
    const int ints_per_w = genoaligner::sw_smem_ints_per_warp(max_n);
    const int block = genoaligner::SW_WARPS_PER_BLOCK * dev_warp;
    const int grid  = (N + genoaligner::SW_WARPS_PER_BLOCK - 1)
                      / genoaligner::SW_WARPS_PER_BLOCK;
    const size_t smem = (size_t)genoaligner::SW_WARPS_PER_BLOCK
                        * (size_t)ints_per_w * sizeof(int);
    printf("  launch  : grid=%d block=%d (%d warps x %d lanes)  smem=%zu B\n\n",
           grid, block, genoaligner::SW_WARPS_PER_BLOCK, dev_warp, smem);

    auto launch = [&] {
        hipLaunchKernelGGL(sw_score_kernel, dim3((unsigned)grid), dim3((unsigned)block),
                           smem, (hipStream_t)0, d_pairs, P, d_res, N, ints_per_w);
    };
    launch();
    hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
        printf("  LAUNCH FAILED: %s\n", hipGetErrorString(le));
        return 1;
    }
    hipDeviceSynchronize();

    // ---- the bench: 7 interleaved reps, every column every rep --------------
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0); hipEventCreate(&ev1);
    launch(); hipDeviceSynchronize();           // warmup, untimed
    parasail_all(1);                            // warmup (also fills ps once)

    std::vector<double> t_gpu, t_gpu_wall, t_p1, t_p8;
    int mismatches = 0;
    printf("-- %d reps, interleaved: gpu(dev) / gpu(wall) / parasail-1t / parasail-8t --\n", reps);
    for (int rep = 0; rep < reps; ++rep) {
        // GPU column
        const auto g0 = std::chrono::steady_clock::now();
        hipEventRecord(ev0);
        launch();
        hipEventRecord(ev1);
        hipDeviceSynchronize();
        const double gwall = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - g0).count();
        float devf = 0.f;
        hipEventElapsedTime(&devf, ev0, ev1);
        t_gpu.push_back(devf); t_gpu_wall.push_back(gwall);

        // every rep, verify -- a fast wrong answer is a FAIL, not a speedup
        hipMemcpy(hr.data(), d_res, (size_t)N * sizeof(SWResult), hipMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i)
            if (hr[(size_t)i].score != ref[(size_t)i]) ++mismatches;

        // parasail single thread
        const auto p0 = std::chrono::steady_clock::now();
        parasail_all(1);
        t_p1.push_back(std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - p0).count());
        for (int i = 0; i < N; ++i)
            if (ps[(size_t)i] != ref[(size_t)i]) ++mismatches;

        // parasail 8 threads
        const auto q0 = std::chrono::steady_clock::now();
        parasail_all(8);
        t_p8.push_back(std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - q0).count());
        for (int i = 0; i < N; ++i)
            if (ps[(size_t)i] != ref[(size_t)i]) ++mismatches;

        printf("  rep %d: gpu=%.3f ms (wall %.3f)  p1=%.1f ms  p8=%.1f ms\n",
               rep, devf, gwall, t_p1.back(), t_p8.back());
    }

    auto line = [&](const char* name, const Stats& s) {
        const double gcups = cells / (s.mean * 1e6);   // ms -> GCUPS
        printf("  %-10s mean=%9.3f ms  sd=%7.3f  min=%9.3f  max=%9.3f"
               "  spread=%5.1f%%   %.2f GCUPS\n",
               name, s.mean, s.sd, s.mn, s.mx,
               100.0 * (s.mx - s.mn) / s.mean, gcups);
        return gcups;
    };
    printf("\n-- summary (cells=%.1f M/rep) --\n", cells / 1e6);
    const double g_gpu = line("gpu-dev",  stats(t_gpu));
    line("gpu-wall", stats(t_gpu_wall));
    const double g_p1 = line("parasail-1t", stats(t_p1));
    const double g_p8 = line("parasail-8t", stats(t_p8));
    printf("\n  gpu-dev vs parasail-1t: %.1fx\n", g_gpu / g_p1);
    printf("  gpu-dev vs parasail-8t: %.1fx\n", g_gpu / g_p8);
    printf("\n  score mismatches across all reps: %d\n", mismatches);

    if (mismatches > 0) {
        printf("\nRESULT: FAIL -- %d wrong scores; no speedup claim survives.\n", mismatches);
        return 1;
    }
    printf("\nRESULT: PASS -- all scores agree with the independent reference;\n"
           "        numbers above are measured, interleaved, on this node.\n");
    return 0;
}
