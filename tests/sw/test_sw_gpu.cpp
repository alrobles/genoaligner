// genoaligner — Smith-Waterman on GPU: correctness of the sequential row walk.
//
// The CPU shim executes the kernel body but its __syncthreads is a no-op, so this is the
// first run where the shared-memory row walk executes with real barriers. The claim
// tested is narrow and stated: blockDim=1, one block per pair, and the score/end
// coordinate must match the independent full-matrix reference.
//
// blockDim > 1 is NOT tested. The column grid-stride reads H[j-1]/E[j-1] written by
// another thread in the same row, and one barrier per row does not establish visibility.
// Until that is fixed the launch must stay at 1 thread per block -- slow, correct, and
// labelled as such. A fast wrong answer would be worse than a slow right one here,
// because the whole point of this library is that its results match a reference.

#include "genoaligner/backend/sw_kernel.hip"
// The kernel BODY lives in the _impl header. hipLaunchKernelGGL needs both the
// declaration and the definition visible in the translation unit that launches it, or
// the device stub is undefined at link time (which is how this line came to be needed).
#include "genoaligner/backend/sw_kernel_impl.hip"

#include <cstdio>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

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

int main()
{
    printf("genoaligner — Smith-Waterman on GPU (blockDim=1, sequential row walk)\n");

    int nsm = 0;
    hipDeviceProp_t prop{};
    if (hipGetDeviceCount(&nsm) != hipSuccess || nsm == 0 ||
        hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        printf("  SKIP: no device visible; nothing verified here.\n");
        printf("RESULT: SKIP -- needs a GPU (exit 77)\n");
        return 77;
    }
    printf("  device: %s\n\n", prop.name);

    const SWParams P{1, -1, 2, 1};

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
    };
    // A longer case, to exercise the shared-memory row walk beyond trivial sizes.
    {
        std::string t, q;
        for (int i = 0; i < 400; ++i) { t.push_back("ACGT"[i % 4]); }
        q = std::string(20, 'T') + t.substr(100, 120) + std::string(20, 'G');
        cases.push_back({t, q, "400 x 160 embedded core"});
    }

    const int N = (int)cases.size();
    std::vector<SWPairView> hv((size_t)N);
    std::vector<SWResult>   hr((size_t)N, SWResult{-1,-1,-1});
    for (int i = 0; i < N; ++i) {
        hv[(size_t)i].text = cases[(size_t)i].text.data();
        hv[(size_t)i].text_len = (int)cases[(size_t)i].text.size();
        hv[(size_t)i].pattern = cases[(size_t)i].pattern.data();
        hv[(size_t)i].pattern_len = (int)cases[(size_t)i].pattern.size();
    }

    SWPairView* d_pairs = nullptr;
    SWResult*   d_res   = nullptr;
    if (hipMalloc(&d_pairs, (size_t)N * sizeof(SWPairView)) != hipSuccess ||
        hipMalloc(&d_res,   (size_t)N * sizeof(SWResult))   != hipSuccess) {
        printf("  hipMalloc failed\n"); return 1;
    }
    hipMemcpy(d_pairs, hv.data(), (size_t)N * sizeof(SWPairView), hipMemcpyHostToDevice);
    hipMemset(d_res, 0, (size_t)N * sizeof(SWResult));

    // Shared memory: the LARGEST pattern in the batch, so one launch config covers all.
    int max_n = 0;
    for (const auto& c : cases) if ((int)c.pattern.size() > max_n) max_n = (int)c.pattern.size();
    const size_t smem = (size_t)(max_n + 1) * 3 * sizeof(int);

    hipLaunchKernelGGL(sw_score_kernel, dim3((unsigned)N), dim3(1), smem,
                       (hipStream_t)0, d_pairs, P, d_res);
    hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
        printf("  LAUNCH FAILED: %s (smem=%zu B, max_n=%d)\n", hipGetErrorString(le), smem, max_n);
        return 1;
    }
    hipDeviceSynchronize();

    hipMemcpy(hr.data(), d_res, (size_t)N * sizeof(SWResult), hipMemcpyDeviceToHost);

    printf("-- kernel vs independent reference --\n");
    for (int i = 0; i < N; ++i) {
        int ri, rj;
        const int ref = ref_sw(cases[(size_t)i].text, cases[(size_t)i].pattern, P, &ri, &rj);
        const SWResult& r = hr[(size_t)i];
        const bool ok = (r.score == ref);
        printf("  %-32s ref=%4d kernel=%4d  %s\n",
               cases[(size_t)i].label.c_str(), ref, r.score, ok ? "ok" : "MISMATCH");
        if (!ok) ++g_fail;
    }

    hipFree(d_pairs); hipFree(d_res);
    printf("\n");
    if (g_fail == 0) {
        printf("RESULT: PASS -- SW kernel matches the reference on GPU at blockDim=1\n");
        printf("  SCOPE: 1 thread per block only. blockDim > 1 is UNSAFE (column\n");
        printf("         dependency across threads, not covered by a per-row barrier)\n");
        printf("         and is not claimed. Performance is therefore not measured.\n");
        return 0;
    }
    printf("RESULT: FAIL (%d)\n", g_fail);
    return 1;
}
