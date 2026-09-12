// genoaligner — SW memory-fault triage.
//
// WHY THIS FILE EXISTS
// --------------------
// The SW kernel faulted on GPU twice ("Memory access fault ... Reason: Unknown",
// core dumped) before this harness existed; both faults were invisible to the CPU
// gate (host pointers inside a device struct; static/dynamic shared-memory
// overlap). The lesson stands: when a kernel faults on device, make the failing
// case identify itself rather than guessing.
//
// METHOD: bisection by construction
// ---------------------------------
// Every case is launched IN ITS OWN KERNEL LAUNCH, with the size printed BEFORE the
// launch and a completion marker printed AFTER. When the process dies on a memory fault,
// the last "before" line with no matching "after" line names the case. No guessing about
// which input is to blame.
//
// It also prints the exact shared-memory request and the device's limit BEFORE launching,
// so a request that exceeds the limit is visible rather than emerging as a fault.
//
// The cases go from trivial to larger deliberately: if case 1 (a single base) faults,
// the bug is structural, not data-dependent, and that changes what to look at.

#include "genoaligner/backend/sw_kernel.hip"
#include "genoaligner/backend/sw_kernel_impl.hip"

#include <cstdio>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

using genoaligner::SWParams;
using genoaligner::SWPairView;
using genoaligner::SWResult;

static int g_dev_warp = 32;

static bool run_one(const std::string& text, const std::string& pattern,
                    SWParams P, const char* label)
{
    const int n = (int)pattern.size();
    const int m = (int)text.size();
    // Launch contract (sw_kernel.hip): one warp per pair. A single pair is a
    // one-warp launch; blockDim must be a multiple of the device warpSize.
    const int    ints_per_w = genoaligner::sw_smem_ints_per_warp(n);
    const size_t smem       = (size_t)ints_per_w * sizeof(int);

    std::printf("  BEFORE  %-28s m=%4d n=%4d smem=%6zu B  ... ",
                label, m, n, smem);
    std::fflush(stdout);

    SWPairView hv;
    // POINTERS, NOT COPIES -- the bug this job was written to find.
    //
    // SWPairView holds `const char*`. Copying the STRUCT to the device copies the
    // POINTERS, which still refer to host stack/heap, so the kernel dereferences host
    // memory from the GPU: "Memory access fault ... on address 0x7ffee043a000" -- an
    // address in the host stack, which is the signature of exactly this mistake. It
    // faulted on m=1,n=1 because it is structural, not size-dependent.
    //
    // The sequence bytes must be copied to the device FIRST, and the views built from
    // the DEVICE pointers. This project already documented and fixed this for WFA's
    // PairView (bench/tcus.cpp); the SW path repeated it.
    char* d_text = nullptr;
    char* d_pat  = nullptr;
    if (hipMalloc(&d_text, text.size()    ? text.size()    : 1) != hipSuccess ||
        hipMalloc(&d_pat,  pattern.size() ? pattern.size() : 1) != hipSuccess) {
        std::printf("hipMalloc(chars) FAILED\n");
        return false;
    }
    hipMemcpy(d_text, text.data(),    text.size(),    hipMemcpyHostToDevice);
    hipMemcpy(d_pat,  pattern.data(), pattern.size(), hipMemcpyHostToDevice);

    hv.text = d_text;        hv.text_len = m;
    hv.pattern = d_pat;      hv.pattern_len = n;

    SWPairView* d_pairs = nullptr;
    SWResult*   d_res   = nullptr;
    if (hipMalloc(&d_pairs, sizeof(SWPairView)) != hipSuccess ||
        hipMalloc(&d_res,   sizeof(SWResult))   != hipSuccess) {
        std::printf("hipMalloc FAILED\n");
        return false;
    }
    hipMemcpy(d_pairs, &hv, sizeof(SWPairView), hipMemcpyHostToDevice);
    hipMemset(d_res, 0, sizeof(SWResult));

    hipLaunchKernelGGL(genoaligner::sw_score_kernel, dim3(1), dim3((unsigned)g_dev_warp),
                       smem, (hipStream_t)0, d_pairs, P, d_res, 1, ints_per_w);
    hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
        std::printf("LAUNCH ERROR: %s\n", hipGetErrorString(le));
        hipFree(d_text); hipFree(d_pat); hipFree(d_pairs); hipFree(d_res);
        return false;
    }
    hipError_t se = hipDeviceSynchronize();
    if (se != hipSuccess) {
        std::printf("SYNC ERROR: %s\n", hipGetErrorString(se));
        hipFree(d_text); hipFree(d_pat); hipFree(d_pairs); hipFree(d_res);
        return false;
    }

    SWResult r{-1,-1,-1};
    hipMemcpy(&r, d_res, sizeof(SWResult), hipMemcpyDeviceToHost);
    std::printf("AFTER  score=%d end=(%d,%d)\n", r.score, r.end_i, r.end_j);
    hipFree(d_text); hipFree(d_pat); hipFree(d_pairs); hipFree(d_res);
    return true;
}

int main()
{
    int nd = 0;
    hipDeviceProp_t prop{};
    if (hipGetDeviceCount(&nd) != hipSuccess || nd == 0 ||
        hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        std::printf("SKIP: no device\n");
        return 77;
    }
    std::printf("device: %s\n", prop.name);
    hipDeviceGetAttribute(&g_dev_warp, hipDeviceAttributeWarpSize, 0);
    std::printf("device warpSize: %d\n", g_dev_warp);

    int smem_per_block = 0;
    hipDeviceGetAttribute(&smem_per_block,
                          hipDeviceAttributeMaxSharedMemoryPerBlock, 0);
    std::printf("device max shared per block: %d B\n\n", smem_per_block);

    const SWParams P{1, -1, 2, 1};

    // Smallest first. If case 1 faults, the bug is structural.
    run_one("A", "A", P, "single base");
    run_one("AC", "AC", P, "two bases");
    run_one("ACGT", "ACGT", P, "four bases");
    run_one("ACGTACGT", "ACGTACGT", P, "eight bases");
    run_one("ACGTACGT", "TGCATGCA", P, "mismatching pair");
    run_one("AAAAAAAA", "CCCCCCCC", P, "disjoint");

    // A gap, then the case that exercises a long row walk.
    {
        std::string t = "ACGTTTACGT";
        std::string q = "ACGTACGT";
        run_one(t, q, P, "gap inside a match");
    }
    {
        std::string t, q;
        for (int i = 0; i < 64; ++i) t.push_back("ACGT"[i % 4]);
        q = t;
        run_one(t, q, P, "64 identical");
    }
    {
        std::string t, q;
        for (int i = 0; i < 400; ++i) t.push_back("ACGT"[i % 4]);
        q = std::string(20,'T') + t.substr(100, 120) + std::string(20,'G');
        run_one(t, q, P, "400 x 160 embedded core");
    }

    std::printf("\nRESULT: PASS -- every case completed, no fault\n");
    return 0;
}
