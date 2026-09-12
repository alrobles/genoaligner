// genoaligner Fase 6 — phase-separated timing probe.
//
// WHY THIS EXISTS BEFORE THE REAL BENCH
// -------------------------------------
// "TCUPS" can mean two very different numbers: the throughput of the kernel
// itself, or the throughput a user sees including allocation and transfer. The
// literature (Accelign, WFA-GPU) reports the first. Reporting the second as if
// it were comparable would be dishonest, and reporting the first without saying
// how much was excluded is the same sin in the other direction.
//
// So this tool measures the SPLIT, on realistic long sequences, before anyone
// commits to a headline number:
//
//     generate cases          (host, once)
//     hipMalloc + hipMemset   (device setup, once)   <- amortised or not?
//     memcpy H2D              (transfer)
//     kernel launch           (hipEvent timed, R repetitions)
//     memcpy D2H              (readback)
//
// The output is milliseconds per phase and, for the kernel, the same work
// expressed as TCUPS so the two framings sit side by side and the decision is
// made on a number rather than a preference.
//
// WHAT "TCUPS" COUNTS HERE
// ------------------------
//   cells = pattern_len * text_len  per pair, summed over pairs.
//   1 TCUPS = 1e12 cells/second.
// This is the same definition the field uses for WFA-style full-DP work, and it
// is deliberately generous to a wavefront method: WFA does NOT visit every cell,
// it visits O(n*s). Reporting cells/ (actual measured time) is therefore a
// fair "what would a full-DP kernel need to do to match this" number when
// compared against full-DP tools, and it is NOT a claim about cells touched.
// Both interpretations are printed so neither can be quoted by accident.
//
// RÉGIME
// ------
// The parity control set is 32-256 bp with adversarial cases -- correct for
// parity, useless for throughput. This uses the phylogenetic regime the project
// targets: identity 50/70/90/95%, lengths up to a few kbp, many pairs.
//
// BUILD (same two backends as the kernel itself):
//   ROCm:  hipcc -O2 -std=c++17 -I. -o tcus bench/tcus.cpp
//   CUDA:  nvcc -w -D__HIP_PLATFORM_NVIDIA__ -x cu ... (see scripts/h6_bench_cuda.sbatch)
//   CPU:   g++ with the shim, for validating this file without a GPU job.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "include/genoaligner/backend/wfa_kernel.hip"
#include "include/genoaligner/backend/wfa_score_flat.hip"
#include "src/reference/edit_distance_cpu.hpp"

using namespace genoaligner;

// ---------------------------------------------------------------------------
// Case generation — same RNG discipline as the parity harness (mt19937 in C++),
// so the sequences here are exactly the kind the kernel was validated on.
// ---------------------------------------------------------------------------
struct Pair {
    std::string text;
    std::string pattern;
};

static std::vector<Pair> build_cases(int n_pairs, int len, int ident_pct, uint32_t seed)
{
    std::vector<Pair> out;
    out.reserve((size_t)n_pairs);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> base(0, 3);
    std::uniform_int_distribution<int> pos(0, len > 1 ? len - 1 : 0);
    std::uniform_int_distribution<int> ch(0, 3);
    const char alphabet[] = "ACGT";

    for (int c = 0; c < n_pairs; ++c) {
        std::string text;
        text.reserve((size_t)len);
        for (int i = 0; i < len; ++i) text.push_back(alphabet[base(rng)]);

        std::string pattern = text;
        const int n_mut = (int)((size_t)len * (100 - ident_pct) / 100.0);
        for (int i = 0; i < n_mut; ++i) pattern[pos(rng)] = alphabet[ch(rng)];

        // Keep indel pressure in, as in the parity set: pure substitutions would
        // let a substitution-only kernel look fast.
        if (c % 7 == 0 && pattern.size() > 4) pattern.erase(0, 2);

        out.push_back({std::move(text), std::move(pattern)});
    }
    return out;
}

static double ms_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv)
{
    int    n_pairs = 2000;
    int    len     = 1024;
    int    smax    = 64;
    int    reps    = 10;
    uint32_t seed  = 12345u;
    int    ident   = 90;
    int    verify  = 25;   // sample pairs scored against the CPU DP reference
    bool   time_gen = false;  // include host case generation in end-to-end?
    bool   use_flat = false;  // run the fixed-block variant instead of the default

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](int def) { return (i + 1 < argc) ? atoi(argv[++i]) : def; };
        if      (a == "--pairs")  n_pairs = next(n_pairs);
        else if (a == "--len")    len     = next(len);
        else if (a == "--ident")  ident   = next(ident);
        else if (a == "--smax")   smax    = next(smax);
        else if (a == "--reps")   reps    = next(reps);
        else if (a == "--verify") verify  = next(verify);
        else if (a == "--time-gen") time_gen = true;
        else if (a == "--kernel" && i + 1 < argc) {
            const std::string k = argv[++i];
            if      (k == "flat") use_flat = true;
            else if (k == "default") use_flat = false;
            else { printf("FATAL: --kernel takes 'flat' or 'default', got '%s'\n", k.c_str()); return 2; }
        }
        else if (a == "--seed")   seed    = (uint32_t)next((int)seed);
    }

    printf("genoaligner Fase 6 -- phase-separated timing probe\n");
    printf("  backend : %s\n", GENOALIGNER_BENCH_BACKEND);
#ifdef __HIP_PLATFORM_NVIDIA__
    printf("  vendor  : NVIDIA (nvcc, __HIP_PLATFORM_NVIDIA__)\n");
#else
    printf("  vendor  : AMD (hipcc)\n");
#endif

    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        printf("  no device -- this probe is a GPU measurement, not the CPU shim.\n");
        return 2;
    }
    printf("  device  : %s\n", prop.name);
    printf("  pairs   : %d   len : %d   ident : %d%%   smax : %d   reps : %d\n\n",
           n_pairs, len, ident, smax, reps);

    // ---- phase 1: host case generation ------------------------------------
    // GENERATED ONCE, OUTSIDE THE MEASUREMENT WINDOW, unless --time-gen is given.
    //
    // The first version timed generation inside end-to-end and it was ~50% of the
    // total -- a property of this benchmark, not of the product. A user does not
    // regenerate their sequences on every alignment call. Keeping it in the
    // headline number would have made the tool look slow for a reason no one
    // shipping it would ever experience.
    //
    // --time-gen restores the old behaviour, so the cost is still measurable for
    // anyone who wants to know what bulk generation costs.
    auto t_gen0 = std::chrono::steady_clock::now();
    std::vector<Pair> cases = build_cases(n_pairs, len, ident, seed);
    const double gen_ms = time_gen ? ms_since(t_gen0) : 0.0;
    if (time_gen) {
        printf("  NOTE: --time-gen is ON; host generation is inside end-to-end.\n\n");
    }
    std::chrono::steady_clock::time_point t0;

    // ---- phase 2: pack into the device layout -----------------------------
    // PairView holds raw POINTERS, not offsets. A memcpy of host-built views
    // would carry host addresses into device code -- a silent wrong-answer bug
    // on unified memory and a fault elsewhere. So the strings are copied first,
    // and the views are built with DEVICE addresses afterwards (below).
    t0 = std::chrono::steady_clock::now();
    std::string text_store, pat_store;
    std::vector<int> toff((size_t)n_pairs), poff((size_t)n_pairs);
    text_store.reserve((size_t)n_pairs * (size_t)len);
    pat_store.reserve((size_t)n_pairs * (size_t)len);
    for (int i = 0; i < n_pairs; ++i) {
        toff[(size_t)i] = (int)text_store.size();
        poff[(size_t)i] = (int)pat_store.size();
        text_store += cases[(size_t)i].text;
        pat_store  += cases[(size_t)i].pattern;
    }
    const double pack_ms = ms_since(t0);

    // ---- phase 2b: device CONTEXT warm-up, measured separately ------------
    // Every run reported memset ~= 460-480 ms REGARDLESS of buffer size (8 KB
    // and 8 MB cost the same). A memset whose cost does not track its size is
    // not a memset: it is the one-time cost of bringing the HIP context up,
    // which lands on whichever device call happens to be first. Attributing it
    // requires doing a throwaway device call BEFORE the timed section and seeing
    // whether the cost moves with it.
    t0 = std::chrono::steady_clock::now();
    {
        void* probe = nullptr;
        hipMalloc(&probe, 4096);
        hipMemset(probe, 0, 4096);
        hipDeviceSynchronize();
        hipFree(probe);
    }
    const double warmup_ms = ms_since(t0);

    // ---- phase 3: device allocation + zeroing -----------------------------
    PairView* d_pairs   = nullptr;
    char*     d_text    = nullptr;
    char*     d_pat     = nullptr;
    int*      d_scores  = nullptr;

    t0 = std::chrono::steady_clock::now();
    hipError_t st = hipMalloc(&d_pairs,  (size_t)n_pairs * sizeof(PairView));
    if (st != hipSuccess) { printf("FATAL: hipMalloc(d_pairs) failed (%s)\n", hipGetErrorString(st)); return 3; }
    const double alloc_pairs_ms = ms_since(t0);

    t0 = std::chrono::steady_clock::now();
    if (st == hipSuccess) st = hipMalloc(&d_text,   text_store.size());
    const double alloc_text_ms = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    if (st == hipSuccess) st = hipMalloc(&d_pat,    pat_store.size());
    const double alloc_pat_ms = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    if (st == hipSuccess) st = hipMalloc(&d_scores, (size_t)n_pairs * sizeof(int));
    const double alloc_sc_ms = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    if (st == hipSuccess) st = hipMemset(d_scores, 0, (size_t)n_pairs * sizeof(int));
    const double memset_ms = ms_since(t0);
    if (st != hipSuccess) {
        printf("FATAL: allocation failed (%s)\n", hipGetErrorString(st));
        return 3;
    }
    const double alloc_ms = alloc_pairs_ms + alloc_text_ms + alloc_pat_ms
                          + alloc_sc_ms + memset_ms;
    printf("  alloc detail: pairs=%.1f text=%.1f pat=%.1f scores=%.1f memset=%.1f ms\n",
           alloc_pairs_ms, alloc_text_ms, alloc_pat_ms, alloc_sc_ms, memset_ms);

    // ---- phase 4: host -> device transfer ---------------------------------
    t0 = std::chrono::steady_clock::now();
    hipMemcpy(d_text,  text_store.data(), text_store.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_pat,   pat_store.data(),  pat_store.size(),  hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    const double h2d_ms = ms_since(t0);

    // Views now point at DEVICE memory. Built after the string copy, never
    // memcpy'd from a host-side struct.
    std::vector<PairView> views((size_t)n_pairs);
    for (int i = 0; i < n_pairs; ++i) {
        views[(size_t)i] = PairView{ d_text + toff[(size_t)i],
                                     (int)cases[(size_t)i].text.size(),
                                     d_pat  + poff[(size_t)i],
                                     (int)cases[(size_t)i].pattern.size(),
                                     0 };
    }
    hipMemcpy(d_pairs, views.data(), (size_t)n_pairs * sizeof(PairView),
              hipMemcpyHostToDevice);
    hipDeviceSynchronize();

    // ---- phase 5: the kernel, timed in isolation --------------------------
    // LAUNCH CONFIG MUST MATCH THE KERNEL'S OWN ASSUMPTIONS.
    //
    // default kernel: maps diagonal k = threadIdx.x - s, so blockDim must be
    // 2*smax+1 (rounded up to a power of two) or it silently misses diagonals.
    //
    // flat kernel: blockDim is DELIBERATELY INDEPENDENT of smax (WFA_FLAT_BLOCK)
    // and the wavefront is swept with an interior grid-stride loop. That is the
    // whole point of the variant: at small real distance the old block was 76-97%
    // idle and paid __syncthreads() over warps doing nothing.
    //
    // Both declare `extern __shared__ int smem[]` sized 2*(2*smax+3) ints; passing 0
    // makes every read land on unallocated memory and the timing measures a faulting
    // guard instead of the algorithm (that was Fase 6 job 29210654).
    const int need_diag = 2 * smax + 1;
    int block = 1;
    while (block < need_diag) block <<= 1;
    if (block > 1024) block = 1024;
    if (use_flat) block = WFA_FLAT_BLOCK;
    const size_t shmem = (size_t)2 * (2 * smax + 3) * sizeof(int);

    printf("  kernel  : %s\n", use_flat ? "flat (fixed block)" : "default (block=2*smax+1)");
    if (use_flat) printf("  WFA_FLAT_BLOCK : %d\n", WFA_FLAT_BLOCK);
    printf("  launch  : block=%d threads, smem=%zu B (%.1f KB)\n",
           block, shmem, shmem / 1024.0);

    // One untimed launch first: the first launch of a kernel pays module
    // loading, and folding that into the average would understate throughput
    // by an amount that depends on how many reps you chose. That is a
    // measurement artefact, not a property of the kernel.
    if (use_flat)
        hipLaunchKernelGGL(wfa_score_kernel_flat, dim3((unsigned)n_pairs), dim3(block), shmem,
                           (hipStream_t)0, d_pairs, d_scores, smax);
    else
        hipLaunchKernelGGL(wfa_score_kernel, dim3((unsigned)n_pairs), dim3(block), shmem,
                           (hipStream_t)0, d_pairs, d_scores, smax);
    hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
        printf("FATAL: launch failed (%s)\n", hipGetErrorString(le));
        return 4;
    }
    hipDeviceSynchronize();

    hipEvent_t ev_start, ev_stop;
    hipEventCreate(&ev_start);
    hipEventCreate(&ev_stop);
    hipEventRecord(ev_start, (hipStream_t)0);
    for (int r = 0; r < reps; ++r) {
        if (use_flat)
            hipLaunchKernelGGL(wfa_score_kernel_flat, dim3((unsigned)n_pairs), dim3(block),
                               shmem, (hipStream_t)0, d_pairs, d_scores, smax);
        else
            hipLaunchKernelGGL(wfa_score_kernel, dim3((unsigned)n_pairs), dim3(block),
                               shmem, (hipStream_t)0, d_pairs, d_scores, smax);
    }
    hipEventRecord(ev_stop, (hipStream_t)0);
    hipEventSynchronize(ev_stop);
    float kernel_ms = 0.0f;
    hipEventElapsedTime(&kernel_ms, ev_start, ev_stop);
    const double per_launch_ms = kernel_ms / (double)reps;

    // ---- phase 6: device -> host readback ---------------------------------
    std::vector<int> scores((size_t)n_pairs, 0);
    t0 = std::chrono::steady_clock::now();
    hipMemcpy(scores.data(), d_scores, (size_t)n_pairs * sizeof(int), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    const double d2h_ms = ms_since(t0);

    // ---- WHAT THE KERNEL ACTUALLY DID -------------------------------------
    // Without this, a large TCUPS is meaningless: the kernel returns -1 for any
    // pair whose true edit distance exceeds smax, WITHOUT visiting its
    // wavefronts. A regime whose pairs are mostly above smax therefore reports a
    // fast time for work it never did, and dividing full-DP cells by that time
    // produces a number that is not a throughput at all.
    int resolved = 0, abandoned = 0;
    for (int i = 0; i < n_pairs; ++i) {
        if (scores[(size_t)i] < 0) ++abandoned; else ++resolved;
    }
    // Solutions carry the score they terminated at, which IS the number of
    // wavefronts the kernel actually walked (it returns at s == distance). The
    // distribution of that value against smax says whether the run paid for the
    // full sweep or stopped at the real distance -- the difference between the
    // two is the single biggest cost in the kernel, so it must be visible.
    int dmax = 0; double dsum = 0.0; int above_smax_half = 0;
    for (int i = 0; i < n_pairs; ++i) {
        const int d = scores[(size_t)i];
        if (d < 0) continue;
        if (d > dmax) dmax = d;
        dsum += d;
        if (d > smax / 2) ++above_smax_half;
    }
    const double dmean = resolved ? dsum / resolved : 0.0;
    // wavefronts walked, averaged over ALL pairs: abandoned ones walk all smax.
    const double wf_used = ((dsum + (double)abandoned * smax) / (double)n_pairs);
    const double wf_offered = (double)smax;
    // Cells actually within the tool's regime: only pairs it resolved.
    double cells_resolved = 0.0;
    for (int i = 0; i < n_pairs; ++i) {
        if (scores[(size_t)i] < 0) continue;
        cells_resolved += (double)views[(size_t)i].pattern_len
                        * (double)views[(size_t)i].text_len;
    }

    // ---- VERIFY: is the kernel doing the work we are timing? --------------
    // The single most important check in this file. A GPU run reported 136 TCUPS
    // while the kernel was returning -1 for every pair, because the launch passed
    // 0 bytes of shared memory and every read landed on unallocated memory. The
    // timing was real; the work was not. Any throughput claim is void without
    // evidence the kernel actually resolved the pairs.
    //
    // Sampled rather than exhaustive: the CPU DP is O(n*m) per pair and this is
    // a timing tool, not the parity harness.
    int v_checked = 0, v_agree = 0, v_skip = 0;
    const int v_n = (verify < n_pairs) ? verify : n_pairs;
    for (int i = 0; i < v_n; ++i) {
        const int got = scores[(size_t)i];
        if (got < 0) { ++v_skip; continue; }   // abandoned: nothing to compare
        const int want = edit_distance_cpu(cases[(size_t)i].pattern.data(),
                                           (int)cases[(size_t)i].pattern.size(),
                                           cases[(size_t)i].text.data(),
                                           (int)cases[(size_t)i].text.size());
        ++v_checked;
        if (got == want) ++v_agree;
        else if (v_checked - v_agree <= 5) {
            printf("  VERIFY MISMATCH pair %d: kernel=%d cpu=%d\n", i, got, want);
        }
    }
    const bool verify_ok = (v_checked > 0) && (v_agree == v_checked);
    printf("VERIFY vs CPU DP (independent O(nm) reference, first %d pairs)\n", v_n);
    printf("  compared : %d   agree : %d   skipped(abandoned) : %d\n",
           v_checked, v_agree, v_skip);
    if (v_checked == 0) {
        printf("  *** NOTHING COMPARED. Either every sampled pair was abandoned or no\n");
        printf("      pair was launched. Any TCUPS below is unverified. ***\n");
    } else if (!verify_ok) {
        printf("  *** KERNEL DISAGREES WITH THE CPU REFERENCE. Timings below are for a\n");
        printf("      WRONG kernel. Do not quote them. ***\n");
    } else {
        printf("  OK: %d/%d sampled pairs match the CPU reference.\n", v_agree, v_checked);
    }
    printf("\n");


    double cells = 0.0;
    for (const auto& pv : views)
        cells += (double)pv.pattern_len * (double)pv.text_len;

    const double kern_s   = per_launch_ms / 1000.0;
    const double setup_ms = gen_ms + pack_ms + alloc_ms + h2d_ms;
    const double e2e_ms   = setup_ms + per_launch_ms + d2h_ms;
    const double tcus_e2e = (e2e_ms > 0.0) ? cells_resolved / (e2e_ms / 1000.0) / 1e12 : 0.0;

    printf("PHASE BREAKDOWN (ms, per full run)\n");
    printf("  context warm-up (1x)    : %10.3f         (one-time per process, NOT in e2e)\n",
           warmup_ms);
    if (time_gen)
        printf("  generate cases (host)   : %10.3f  %5.1f%%\n", gen_ms, 100.0 * gen_ms / e2e_ms);
    else
        printf("  generate cases (host)   : %10.3f         (excluded; --time-gen to include)\n",
               gen_ms);
    printf("  pack to device layout   : %10.3f  %5.1f%%\n", pack_ms, 100.0 * pack_ms / e2e_ms);
    printf("  hipMalloc + memset      : %10.3f  %5.1f%%\n", alloc_ms, 100.0 * alloc_ms / e2e_ms);
    printf("  memcpy H2D              : %10.3f  %5.1f%%\n", h2d_ms, 100.0 * h2d_ms / e2e_ms);
    printf("  KERNEL (mean of %d)     : %10.3f  %5.1f%%   <-- the literature number\n",
           reps, per_launch_ms, 100.0 * per_launch_ms / e2e_ms);
    printf("  memcpy D2H              : %10.3f  %5.1f%%\n", d2h_ms, 100.0 * d2h_ms / e2e_ms);
    printf("  ---------------------------------------------\n");
    printf("  end-to-end              : %10.3f\n\n", e2e_ms);

    printf("THROUGHPUT (1 TCUPS = 1e12 cells/s)\n");
    printf("  cells per run           : %.3e  (%d pairs x %d x %d)\n",
           cells, n_pairs, len, len);
    printf("  resolved / abandoned    : %d / %d   (abandoned = true distance > smax=%d)\n",
           resolved, abandoned, smax);
    printf("  distance (score) mean/max: %.1f / %d      <-- wavefronts actually walked\n",
           dmean, dmax);
    printf("  wavefronts walked/offered: %.1f / %d   (%.1f%% of the smax sweep)\n",
           wf_used, smax, 100.0 * wf_used / wf_offered);
    if (wf_used < 0.5 * wf_offered)
        printf("  ^ ** the kernel walks the full sweep to smax but the real distance is\n"
               "      far smaller: most of the barrier traffic is over empty wavefronts. **\n");
    printf("  pairs with d > smax/2   : %d\n", above_smax_half);
    printf("  cells actually resolved : %.3e  (%.1f%% of the naive cell count)\n",
           cells_resolved, cells > 0.0 ? 100.0 * cells_resolved / cells : 0.0);
    if (abandoned == n_pairs) {
        printf("  *** EVERY PAIR WAS ABANDONED. The kernel returned early for all of\n");
        printf("      them, so the timings below measure the guard, not the algorithm.\n");
        printf("      Any TCUPS printed here is not a throughput. Lower the identity\n");
        printf("      overlap or raise smax until pairs are actually resolved. ***\n");
    } else if (abandoned > 0) {
        printf("  NOTE: %d/%d pairs abandoned; the TCUPS below is computed over the\n",
               abandoned, n_pairs);
        printf("      CELLS OF RESOLVED PAIRS ONLY, so it is not diluted by pairs the\n");
        printf("      kernel skipped. Kernel time still includes the skipped pairs'\n");
        printf("      guard cost, which makes this a conservative estimate.\n");
    }
    printf("  kernel-only ............ : %8.3f TCUPS   (comparable to Accelign/WFA-GPU)\n",
           (kern_s > 0.0) ? cells_resolved / kern_s / 1e12 : 0.0);
    printf("  end-to-end ............. : %8.3f TCUPS   (what a user sees; NOT comparable)\n",
           tcus_e2e);
    printf("  setup share of e2e ..... : %5.1f%%\n", 100.0 * setup_ms / e2e_ms);

    hipFree(d_pairs); hipFree(d_text); hipFree(d_pat); hipFree(d_scores);
    hipEventDestroy(ev_start); hipEventDestroy(ev_stop);

    // Exit code carries the verdict: a timing run whose kernel did not match the
    // CPU reference (or did nothing) must not look like a successful benchmark.
    if (v_checked == 0) return 5;
    if (!verify_ok)     return 6;
    return 0;
}
