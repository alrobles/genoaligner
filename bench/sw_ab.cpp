// genoaligner — SW speedup measurement: sequential (blockDim=1) vs warp-per-row.
//
// METHODOLOGY IS B6's, ON PURPOSE (docs/RESULTADO_B6_TCUPS.md):
//   * everything inside ONE job -- no numbers are ever compared across jobs;
//   * 7 independent reps; BOTH kernels inside every rep, interleaved A then B,
//     so intra-job drift (clocks, neighbours) hits both equally;
//   * device time via hipEvent, plus wall time reported for context -- this
//     cluster time-slices GPUs across jobs (shard:mi210 gres, see
//     docs/RESULTADO_H8_SW1_WARP.md §3), so a stalled rep shows up in BOTH
//     kernels' columns, which is what makes the interleave meaningful;
//   * reported as mean +/- sd, min, max, spread -- never a bare ratio.
//
// Kernel A is the ba795db version (one block per pair, ONE thread per block,
// shared-memory row buffers) embedded verbatim below and renamed
// sw_score_kernel_v1. Kernel B is the shipped warp-per-row kernel.

#include "genoaligner/backend/sw_kernel.hip"
#include "genoaligner/backend/sw_kernel_impl.hip"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include <random>

namespace genoaligner { namespace v1 {

// ---- kernel A: the verified blockDim=1 implementation, verbatim from ba795db
// ---- (renamed; body identical so the comparison is kernel-vs-kernel, not
// ---- kernel-vs-reimplementation). SW_BLOCK was its fixed block size constant.
static constexpr int SW_BLOCK_V1 = 128;

__global__ void sw_score_kernel_v1(const SWPairView* __restrict__ pairs,
                                   SWParams           params,
                                   SWResult* __restrict__ out)
{
    const int pair_id = blockIdx.x;
    const SWPairView p = pairs[pair_id];

    const int m = p.text_len;
    const int n = p.pattern_len;

    if (m <= 0 || n <= 0) {
        if (threadIdx.x == 0) out[pair_id] = SWResult{0, -1, -1};
        return;
    }

    extern __shared__ int smem[];
    int* H = smem;              // (n+1)
    int* E = H + (n + 1);       // (n+1)
    int* F = E + (n + 1);       // (n+1)

    for (int j = threadIdx.x; j <= n; j += blockDim.x) {
        H[j] = 0; E[j] = 0; F[j] = 0;
    }
    __syncthreads();

    int  my_best = 0;
    int  my_bi = -1;
    int  my_bj = -1;

    __syncthreads();

    for (int i = 1; i <= m; ++i) {
        const char a = p.text[i - 1];
        int diag_prev = 0;

        for (int j = (int)threadIdx.x + 1; j <= n; j += (int)blockDim.x) {
            const char b = p.pattern[j - 1];
            const int  s = (a == b) ? params.match : params.mismatch;

            int e = E[j - 1] - params.gap_extend;
            const int e_open = H[j - 1] - params.gap_open;
            if (e_open > e) e = e_open;

            int f = F[j] - params.gap_extend;
            const int f_open = H[j] - params.gap_open;
            if (f_open > f) f = f_open;

            int h = diag_prev + s;
            if (e > h) h = e;
            if (f > h) h = f;
            if (h < 0) h = 0;

            if (h > my_best) { my_best = h; my_bi = i - 1; my_bj = j - 1; }

            const int h_diag_next = H[j];
            H[j] = h;
            E[j] = e;
            F[j] = f;
            diag_prev = h_diag_next;
        }
        __syncthreads();
    }

    int* r_best = smem + 3 * (n + 1);
    int* r_i    = r_best + SW_BLOCK_V1;
    int* r_j    = r_i + SW_BLOCK_V1;

    r_best[threadIdx.x] = my_best;
    r_i[threadIdx.x]    = my_bi;
    r_j[threadIdx.x]    = my_bj;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            if (r_best[threadIdx.x + stride] > r_best[threadIdx.x]) {
                r_best[threadIdx.x] = r_best[threadIdx.x + stride];
                r_i[threadIdx.x]    = r_i[threadIdx.x + stride];
                r_j[threadIdx.x]    = r_j[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) out[pair_id] = SWResult{r_best[0], r_i[0], r_j[0]};
}

}}  // namespace genoaligner::v1

using genoaligner::SWParams;
using genoaligner::SWPairView;
using genoaligner::SWResult;

struct Case { std::string text, pattern, label; };

int main()
{
    printf("genoaligner — SW speedup: sequential vs warp-per-row (interleaved, B6 design)\n");

    int nsm = 0;
    hipDeviceProp_t prop{};
    if (hipGetDeviceCount(&nsm) != hipSuccess || nsm == 0 ||
        hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        printf("  SKIP: no device visible (exit 77)\n");
        return 77;
    }
    int dev_warp = 0;
    hipDeviceGetAttribute(&dev_warp, hipDeviceAttributeWarpSize, 0);
    printf("  device: %s   warpSize=%d\n", prop.name, dev_warp);
    printf("  ROCR_VISIBLE_DEVICES=%s\n\n",
           getenv("ROCR_VISIBLE_DEVICES") ? getenv("ROCR_VISIBLE_DEVICES") : "<unset>");

    const SWParams P{1, -1, 2, 1};

    // Workload: the same shape as the correctness batch plus a heavier tail, so the
    // sequential kernel has enough work for the difference to be measurable without
    // making the job long. Fixed seed -> identical bytes for both kernels.
    std::vector<Case> cases;
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
        // The long case from the correctness suite: dominates the sequential kernel.
        std::string t, q;
        for (int i = 0; i < 400; ++i) t.push_back("ACGT"[i % 4]);
        q = std::string(20, 'T') + t.substr(100, 120) + std::string(20, 'G');
        cases.push_back({t, q, "400 x 160 embedded core"});
        cases.push_back({std::string(300, 'A'), std::string(300, 'A'), "300 identical"});
        cases.push_back({std::string(512,'A')+"CGTACGTACGTACGTACGT", 
                         std::string(512,'T')+"CGTACGTACGTACGTACGT", "531 flank-aligned"});
    }
    const int N = (int)cases.size();

    std::vector<SWPairView> hv((size_t)N);
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
    SWResult *d_resA = nullptr, *d_resB = nullptr;
    if (hipMalloc(&d_pairs, (size_t)N * sizeof(SWPairView)) != hipSuccess ||
        hipMalloc(&d_resA,  (size_t)N * sizeof(SWResult))   != hipSuccess ||
        hipMalloc(&d_resB,  (size_t)N * sizeof(SWResult))   != hipSuccess) {
        printf("  hipMalloc failed\n"); return 1;
    }
    hipMemcpy(d_pairs, hv.data(), (size_t)N * sizeof(SWPairView), hipMemcpyHostToDevice);
    hipMemset(d_resA, 0, (size_t)N * sizeof(SWResult));
    hipMemset(d_resB, 0, (size_t)N * sizeof(SWResult));

    int max_n = 0;
    for (const auto& c : cases) if ((int)c.pattern.size() > max_n) max_n = (int)c.pattern.size();

    // Launch A: the v1 contract -- one block per pair, ONE thread, smem sized by the
    // largest pattern plus the SW_BLOCK reduction tail.
    const size_t smemA = ((size_t)(max_n + 1) * 3 + 3 * genoaligner::v1::SW_BLOCK_V1)
                         * sizeof(int);
    auto launchA = [&] {
        hipLaunchKernelGGL(genoaligner::v1::sw_score_kernel_v1,
                           dim3((unsigned)N), dim3(1), smemA, (hipStream_t)0,
                           d_pairs, P, d_resA);
    };

    // Launch B: the shipped contract -- one warp per pair, warps packed per block.
    const int ints_per_w = genoaligner::sw_smem_ints_per_warp(max_n);
    const int blockB     = genoaligner::SW_WARPS_PER_BLOCK * dev_warp;
    const int gridB      = (N + genoaligner::SW_WARPS_PER_BLOCK - 1)
                           / genoaligner::SW_WARPS_PER_BLOCK;
    const size_t smemB   = (size_t)genoaligner::SW_WARPS_PER_BLOCK
                           * (size_t)ints_per_w * sizeof(int);
    auto launchB = [&] {
        hipLaunchKernelGGL(genoaligner::sw_score_kernel,
                           dim3((unsigned)gridB), dim3((unsigned)blockB), smemB,
                           (hipStream_t)0, d_pairs, P, d_resB, N, ints_per_w);
    };

    printf("  A (seq):  grid=%d block=1   smem=%zu B\n", N, smemA);
    printf("  B (warp): grid=%d block=%d  smem=%zu B\n\n", gridB, blockB, smemB);

    launchA(); launchB(); hipDeviceSynchronize();
    hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
        printf("  LAUNCH FAILED: %s\n", hipGetErrorString(le)); return 1;
    }

    // A/B outputs must agree, or the timing comparison is meaningless.
    std::vector<SWResult> hA((size_t)N), hB((size_t)N);
    hipMemcpy(hA.data(), d_resA, (size_t)N * sizeof(SWResult), hipMemcpyDeviceToHost);
    hipMemcpy(hB.data(), d_resB, (size_t)N * sizeof(SWResult), hipMemcpyDeviceToHost);
    int disagree = 0;
    for (int i = 0; i < N; ++i) {
        if (hA[(size_t)i].score != hB[(size_t)i].score) {
            if (disagree < 8)
                printf("  DISAGREE pair %d (%s): A=%d B=%d\n", i,
                       cases[(size_t)i].label.c_str(),
                       hA[(size_t)i].score, hB[(size_t)i].score);
            ++disagree;
        }
    }
    printf("  A vs B outputs: %d/%d agree\n", N - disagree, N);
    if (disagree) { printf("RESULT: FAIL (kernels disagree)\n"); return 1; }

    // Interleaved timing: inside every rep, A then B. 7 reps.
    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    launchA(); hipDeviceSynchronize();      // warmups, untimed
    launchB(); hipDeviceSynchronize();

    printf("\n-- 7 reps, interleaved A,B (dev = hipEvent ms) --\n");
    double a[7], b[7], aw[7], bw[7];
    for (int rep = 0; rep < 7; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        hipEventRecord(e0); launchA(); hipEventRecord(e1);
        hipDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        float fa = 0; hipEventElapsedTime(&fa, e0, e1);
        aw[rep] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        a[rep]  = (double)fa;

        const auto t2 = std::chrono::steady_clock::now();
        hipEventRecord(e0); launchB(); hipEventRecord(e1);
        hipDeviceSynchronize();
        const auto t3 = std::chrono::steady_clock::now();
        float fb = 0; hipEventElapsedTime(&fb, e0, e1);
        bw[rep] = std::chrono::duration<double, std::milli>(t3 - t2).count();
        b[rep]  = (double)fb;

        printf("  rep %d: A dev=%8.3f wall=%8.3f | B dev=%8.3f wall=%8.3f\n",
               rep, a[rep], aw[rep], b[rep], bw[rep]);
    }
    hipEventDestroy(e0); hipEventDestroy(e1);

    auto stats = [](const double* x) {
        double mn = 1e30, mx = 0, sum = 0, sq = 0;
        for (int i = 0; i < 7; ++i) {
            if (x[i] < mn) mn = x[i];
            if (x[i] > mx) mx = x[i];
            sum += x[i]; sq += x[i] * x[i];
        }
        const double mean = sum / 7;
        const double sd = std::sqrt(sq / 7 - mean * mean);
        const double sp = (mn > 0) ? (mx - mn) / mn : 0;
        struct R { double mean, sd, mn, mx, sp; };
        return R{mean, sd, mn, mx, sp};
    };
    auto sa = stats(a), sb = stats(b);
    auto wa = stats(aw), wb = stats(bw);

    printf("\n  A dev : mean=%.3f sd=%.3f min=%.3f max=%.3f spread=%.1f%%\n",
           sa.mean, sa.sd, sa.mn, sa.mx, 100.0 * sa.sp);
    printf("  B dev : mean=%.3f sd=%.3f min=%.3f max=%.3f spread=%.1f%%\n",
           sb.mean, sb.sd, sb.mn, sb.mx, 100.0 * sb.sp);
    printf("  A wall: mean=%.3f spread=%.1f%%   B wall: mean=%.3f spread=%.1f%%\n",
           wa.mean, 100.0 * wa.sp, wb.mean, 100.0 * wb.sp);
    printf("  speedup (A/B, mean of dev): %.2fx  (min/min: %.2fx, max/max: %.2fx)\n",
           sa.mean / sb.mean, sa.mn / sb.mn, sa.mx / sb.mx);

    // Gate rule for THIS measurement (not a kernel verdict): if EITHER kernel's dev
    // spread exceeds 25% the co-tenant hypothesis is live and the ratio is flagged,
    // not hidden.
    const bool noisy = (sa.sp > 0.25 || sb.sp > 0.25);
    printf("\nRESULT: %s -- %s\n", noisy ? "NOISY" : "PASS",
           noisy ? "interleaved spreads too wide; ratio reported but flagged"
                 : "both kernels stable inside the job; ratio is meaningful");
    return noisy ? 2 : 0;
}
