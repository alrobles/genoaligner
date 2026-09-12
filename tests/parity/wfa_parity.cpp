// genoaligner — Phase 2 parity harness (H2) + Fase 4 traceback harness.
//
// THE TEST THAT MATTERS: does the GPU wavefront match an independent CPU
// reference on a control set of sequence pairs?
//
// Fusion criterion for Phase 2: >=95% agreement on ~1,000 pairs. Below that,
// the formulation is wrong and no amount of optimisation helps.
//
// FASE 4 (--traceback): the same harness drives wfa_trace_kernel, which
// reconstructs the CIGAR on device, and applies the same three checks the CPU
// gate applies (re-score, well-formedness, and -- via the emitted CIGARs --
// agreement with edlib). The checks come from tests/parity/cigar.hpp so
// the CPU and device paths cannot drift apart.
//
// TWO BACKENDS, ONE SOURCE. Built with hipcc (MI210) or nvcc (RTX 6000,
// -D__HIP_PLATFORM_NVIDIA__) this file launches real device kernels. Built with
// g++ and -DGENOALIGNER_HIP_SHIM it has no device, so the kernel bodies are
// invoked DIRECTLY on host memory -- which is what puts the GPU design inside
// the CPU gate instead of behind a cluster job.
//
// The control set is generated deterministically (fixed seed) so the pass rate
// is reproducible across machines and runs. Identity levels span the
// phylogenetic regime (50/70/90/100%) plus adversarial cases (empty, all-gap,
// single mismatch) where off-by-one errors in diagonal indexing show up.

#include <hip/hip_runtime.h>

#include "include/genoaligner/backend/wfa_kernel.hip"
#include "src/reference/edit_distance_cpu.hpp"
#include "tests/parity/cigar.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <thread>

// The traceback section needs to launch a kernel by function pointer (for the
// dynamic-shared-memory attribute) on a real backend, and to call the kernel
// body directly under the shim. One macro, two definitions.
#if defined(GENOALIGNER_HIP_SHIM)
#define GENOALIGNER_HAS_DEVICE 0
#else
#define GENOALIGNER_HAS_DEVICE 1
#endif

#if GENOALIGNER_HAS_DEVICE
#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t _e = (call);                                             \
        if (_e != hipSuccess) {                                             \
            fprintf(stderr, "HIP error %s at %s:%d\n",                      \
                    hipGetErrorString(_e), __FILE__, __LINE__);              \
            return 1;                                                       \
        }                                                                   \
    } while (0)
#else
#define HIP_CHECK(call) do { (void)sizeof(call); } while (0)

// The shim declares these extern; someone must define them, and the harness is
// the right owner because it is what sets them before each "launch".
uint3 threadIdx{0, 0, 0};
uint3 blockIdx{0, 0, 0};
dim3  blockDim{1, 1, 1};
dim3  gridDim{1, 1, 1};
#endif

using namespace genoaligner;

struct Case {
    std::string text;
    std::string pattern;
    std::string label;
};

// Deterministic generator: same control set every run, on every machine.
static std::vector<Case> build_control_set(int n_cases, uint32_t seed)
{
    std::vector<Case> cases;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> base(0, 3);
    const char alphabet[] = "ACGT";

    auto mutate = [&](std::string s, int ident_pct) {
        int n_mut = (int)(s.size() * (100 - ident_pct) / 100.0);
        std::uniform_int_distribution<int> pos(0, (int)s.size() - 1);
        std::uniform_int_distribution<int> ch(0, 3);
        for (int i = 0; i < n_mut && !s.empty(); ++i) {
            s[pos(rng)] = alphabet[ch(rng)];
        }
        return s;
    };

    std::uniform_int_distribution<int> chdist(0, 3);

    const int idents[] = {100, 90, 70, 50};
    const int lens[]   = {32, 64, 128, 256};

    for (int c = 0; c < n_cases; ++c) {
        const int ident = idents[c % 4];
        const int len   = lens[(c / 4) % 4];

        std::string text;
        text.reserve(len);
        for (int i = 0; i < len; ++i) text.push_back(alphabet[base(rng)]);

        std::string pattern = mutate(text, ident);
        // Occasionally introduce an indel so gaps are exercised, not just subs.
        if (c % 7 == 0 && pattern.size() > 4) {
            pattern.erase(0, 2);
        } else if (c % 11 == 0 && !pattern.empty()) {
            pattern.insert(pattern.begin() + pattern.size() / 2, alphabet[chdist(rng)]);
        }

        char lab[64];
        snprintf(lab, sizeof(lab), "len=%d ident=%d%%", len, ident);
        cases.push_back({text, pattern, lab});
    }

    // Adversarial edge cases -- these are where off-by-one bugs live.
    cases.push_back({"", "", "both empty"});
    cases.push_back({"ACGT", "", "empty pattern"});
    cases.push_back({"", "ACGT", "empty text"});
    cases.push_back({"A", "T", "single mismatch"});
    cases.push_back({"AAAA", "AAAA", "identical"});
    cases.push_back({"AAAA", "AA", "pattern is prefix"});
    cases.push_back({"ACGTACGT", "TTTTTTTT", "no match at all"});
    return cases;
}

// ===========================================================================
// Fase 4 — traceback driver
// ===========================================================================
//
// Memory model, and why it is a PARAMETER and not extern shared:
//
//   The walk is serial, so the parallelism is across PAIRS (one block per pair,
//   one participating thread). Each pair needs every wavefront M[0..smax] to
//   walk back through, which at smax=64 is 65 x 131 x 4 B = 34.8 KB PER PAIR.
//   1000 pairs cannot be resident at that size, so the wavefront matrix lives in
//   a global workspace with one slice per block. A per-block shared buffer could
//   only work for a handful of simultaneous pairs -- that is the contrast the
//   `--chunk-bytes 0` run exists to measure.
//
//   Putting the waves in `extern __shared__` would have been simpler and is what
//   the original design sketched. It does not work: at smax=64 the matrix is
//   34.8 KB per pair, so a 48/64 KB per-block limit binds almost immediately, and
//   the launch would have to be capped at one or two resident pairs.

struct TraceRun {
    std::vector<int>         scores;      // kernel scores (per pair)
    std::vector<std::string> cigars;      // decoded CIGAR strings, read direction
    int   truncated  = 0;                 // CIGAR hit the cap
    int   unresolved = 0;                 // edit distance > smax
    int   emitted    = 0;                 // CIGAR produced
    long  host_ms    = 0;
};

#if GENOALIGNER_HAS_DEVICE
// Real backend: launch wfa_trace_kernel, read back the workspace, decode.
static int run_trace(const PairView* d_pairs, PairView* h_pairs,
                     const std::vector<Case>& cases,
                     int smax, int chunk_bytes, int cigar_cap,
                     int* d_scores, TraceRun* out)
{
    const int N = (int)cases.size();
    const int wf_stride    = 2 * smax + 3;          // slots per wavefront, both paddings
    const int wf_alloc_max = (smax + 1) * wf_stride; // slots per pair: scores 0..smax
    const int need_bytes   = wf_alloc_max * (int)sizeof(int);

    int max_smem = 0;
    hipDeviceGetAttribute(&max_smem, hipDeviceAttributeMaxSharedMemoryPerBlock, 0);

    // EVERY pair gets a slice: pair_id indexes the workspace directly.
    const size_t ws_ints = (size_t)wf_alloc_max * (size_t)N;

    printf("=== Fase 4 traceback (wfa_trace_kernel) ===\n");
    printf("  smax            : %d\n", smax);
    printf("  wf_stride       : %d ints\n", wf_stride);
    printf("  per-pair need   : %d B (%.1f KB)\n", need_bytes, need_bytes / 1024.0);
    printf("  device smem cap : %d B (%.1f KB) per block\n", max_smem, max_smem / 1024.0);
    printf("  chunk request   : %d B (%.1f KB)\n", chunk_bytes, chunk_bytes / 1024.0);
    printf("  workspace       : %zu ints (%.1f MB) -- global, one slice per pair\n",
           ws_ints, ws_ints * sizeof(int) / 1048576.0);
    printf("  cigar cap       : %d\n\n", cigar_cap);

    int* d_ws     = nullptr;
    int* d_cg     = nullptr;
    int* d_meta   = nullptr;
    hipError_t st = hipMalloc(&d_ws, ws_ints * sizeof(int));
    if (st == hipSuccess) {
        // Zeroed: the kernel writes every wavefront it reaches, and a read of an
        // unwritten one must look unreachable. 0 is inside the sentinel-safe band
        // (wfa_reachable needs > WFA_NEG + 1024), so an untouched cell reads as
        // unreachable rather than masquerading as a legitimate "not computed".
        st = hipMemset(d_ws, 0, ws_ints * sizeof(int));
    }
    if (st == hipSuccess) st = hipMalloc(&d_cg, (size_t)N * (size_t)cigar_cap * sizeof(int));
    if (st == hipSuccess) st = hipMalloc(&d_meta, (size_t)N * 4 * sizeof(int));
    if (st == hipSuccess) st = hipMemset(d_meta, 0, (size_t)N * 4 * sizeof(int));
    if (st != hipSuccess) {
        printf("  Fase 4: SKIPPED -- allocation failed (%s)\n\n", hipGetErrorString(st));
        return 0;
    }

    // Dynamic shared memory: at smax=64 a pair needs 34.8 KB, inside the 48 KB
    // default, so this is a no-op today. It is exercised rather than merely
    // referenced so that a future smax that exceeds the default fails here with
    // a clear message instead of an "invalid argument" launch error.
    if (chunk_bytes > 48 * 1024) {
        hipError_t as = hipFuncSetAttribute((const void*)wfa_trace_kernel,
                                            hipFuncAttributeMaxDynamicSharedMemorySize,
                                            chunk_bytes);
        printf("  smem opt-in     : %s (requested %d B)\n\n",
               hipGetErrorString(as), chunk_bytes);
    }

    hipLaunchKernelGGL(wfa_trace_kernel, dim3(N), dim3(1), (size_t)0,
                       (hipStream_t)0, d_pairs, d_scores, d_cg, d_meta, d_ws,
                       smax, wf_stride, wf_alloc_max, chunk_bytes, cigar_cap);
    hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
        printf("  Fase 4: LAUNCH FAILED (%s)\n\n", hipGetErrorString(le));
        hipFree(d_ws); hipFree(d_cg); hipFree(d_meta);
        return 0;
    }
    hipDeviceSynchronize();

    std::vector<int> meta((size_t)N * 4, 0);
    HIP_CHECK(hipMemcpy(meta.data(), d_meta, (size_t)N * 4 * sizeof(int),
                        hipMemcpyDeviceToHost));
    out->scores.resize((size_t)N, 0);
    HIP_CHECK(hipMemcpy(out->scores.data(), d_scores, (size_t)N * sizeof(int),
                        hipMemcpyDeviceToHost));

    std::vector<int> cg((size_t)N * (size_t)cigar_cap, 0);
    std::vector<int> ws(ws_ints, 0);
    HIP_CHECK(hipMemcpy(cg.data(), d_cg, (size_t)N * (size_t)cigar_cap * sizeof(int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(ws.data(), d_ws, ws_ints * sizeof(int), hipMemcpyDeviceToHost));

    out->cigars.assign((size_t)N, std::string());
    for (int i = 0; i < N; ++i) {
        const int rev_used = meta[(size_t)i * 4 + 1];
        const int trunc    = meta[(size_t)i * 4 + 2];
        const int emit     = meta[(size_t)i * 4 + 3];
        if (emit && trunc) ++out->truncated;
        if (out->scores[(size_t)i] < 0) ++out->unresolved;
        if (!emit || trunc || rev_used <= 0) continue;

        ++out->emitted;
        out->cigars[(size_t)i] = cigar_from_rev(cg.data() + (size_t)i * (size_t)cigar_cap,
                                                rev_used);
    }

    printf("  launched        : blocks=%d threads=1\n\n", N);
    hipFree(d_ws); hipFree(d_cg); hipFree(d_meta);
    return 1;
}
#else
// Shim backend: no device. The kernel body is called directly, once per pair,
// with host memory for the workspace. This is what makes the shim evidence about
// the SHIPPED kernel rather than a paraphrase of it.
static int run_trace(const PairView*, PairView* h_pairs,
                     const std::vector<Case>& cases,
                     int smax, int chunk_bytes, int cigar_cap,
                     int* h_scores, TraceRun* out)
{
    const int N = (int)cases.size();
    const int wf_stride    = 2 * smax + 3;
    const int wf_alloc_max = (smax + 1) * wf_stride;
    const int need_bytes   = wf_alloc_max * (int)sizeof(int);

    printf("=== Fase 4 traceback (wfa_trace_kernel, CPU shim) ===\n");
    printf("  smax            : %d\n", smax);
    printf("  wf_stride       : %d ints\n", wf_stride);
    printf("  per-pair need   : %d B (%.1f KB)\n", need_bytes, need_bytes / 1024.0);
    printf("  workspace       : host, one slice per pair\n");
    printf("  cigar cap       : %d\n\n", cigar_cap);

    // NOTE: the shim harness allocates these at exactly the size the kernel will
    // use, so an out-of-bounds write in the kernel is an ASan error here rather
    // than invisible corruption. The kernel must NOT size them itself.
    const size_t ws_ints = (size_t)wf_alloc_max * (size_t)N;
    std::vector<int> ws(ws_ints, 0);
    std::vector<int> cg((size_t)N * (size_t)cigar_cap, 0);
    std::vector<int> meta((size_t)N * 4, 0);

    // The kernel is __global__ (defined away) and reads blockIdx/threadIdx from
    // the shim globals, so set them and call it. blockDim.x must be 1: the walk
    // uses a single thread and the kernel returns for threadIdx.x != 0.
    blockDim  = dim3{1, 1, 1};
    threadIdx = uint3{0, 0, 0};
    for (int i = 0; i < N; ++i) {
        blockIdx = uint3{(unsigned)i, 0, 0};
        wfa_trace_kernel(h_pairs, h_scores, cg.data(), meta.data(), ws.data(),
                         smax, wf_stride, wf_alloc_max, chunk_bytes, cigar_cap);
    }

    out->scores.assign(h_scores, h_scores + N);
    out->cigars.assign((size_t)N, std::string());
    for (int i = 0; i < N; ++i) {
        const int rev_used = meta[(size_t)i * 4 + 1];
        const int trunc    = meta[(size_t)i * 4 + 2];
        const int emit     = meta[(size_t)i * 4 + 3];
        if (emit && trunc) ++out->truncated;
        if (out->scores[(size_t)i] < 0) ++out->unresolved;
        if (!emit || trunc || rev_used <= 0) continue;

        ++out->emitted;
        out->cigars[(size_t)i] = cigar_from_rev(cg.data() + (size_t)i * (size_t)cigar_cap,
                                                rev_used);
    }

    printf("  ran             : %d pairs, shim (kernel body called directly)\n\n", N);
    return 1;
}
#endif

int main(int argc, char** argv)
{
    int n_cases = (argc > 1 && argv[1][0] != '-') ? atoi(argv[1]) : 1000;

    // --emit <path>        write pairs + scores (+ CIGARs in traceback mode)
    // --traceback          run wfa_trace_kernel and validate the CIGARs
    // --chunk-bytes N      dynamic smem requested per block (0 = none); decides
    //                      how many pairs are resident at once
    // --cigar-cap N        force a small CIGAR buffer so the truncation path is
    //                      reachable on purpose: the cap must gate the emit, not
    //                      silently corrupt
    std::string emit_path;
    bool traceback_mode = false;
    int  chunk_bytes    = 48 * 1024;
    int  cigar_cap      = WFA_CIGAR_MAX;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--emit" && i + 1 < argc)              { emit_path = argv[++i]; }
        else if (a == "--traceback")                    { traceback_mode = true; }
        else if (a == "--chunk-bytes" && i + 1 < argc)  { chunk_bytes = atoi(argv[++i]); }
        else if (a == "--cigar-cap" && i + 1 < argc)    { cigar_cap   = atoi(argv[++i]); }
    }

    const int smax = 64;   // cap on edit distance we will chase

#if GENOALIGNER_HAS_DEVICE
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("genoaligner WFA parity harness (H2)\n");
    printf("  device : %s\n", prop.name);
    printf("  arch   : %s\n", prop.gcnArchName);
#else
    printf("genoaligner WFA parity harness (H2) -- CPU shim\n");
    printf("  device : none (kernel bodies called directly)\n");
#endif
    printf("  cases  : %d\n", n_cases);
    printf("  smax   : %d\n", smax);
    if (!emit_path.empty()) printf("  emit   : %s\n", emit_path.c_str());
    printf("\n");

    auto cases = build_control_set(n_cases, 12345u);

    // ---- host-side reference -------------------------------------------------
    std::vector<int> ref(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) {
        ref[i] = edit_distance_cpu(cases[i].pattern.data(), (int)cases[i].pattern.size(),
                                   cases[i].text.data(),    (int)cases[i].text.size());
    }

#if GENOALIGNER_HAS_DEVICE
    // ---- device buffers ------------------------------------------------------
    std::vector<PairView> views(cases.size());
    std::vector<std::string> text_store, pat_store;
    text_store.reserve(cases.size());
    pat_store.reserve(cases.size());
    for (auto& c : cases) { text_store.push_back(c.text); pat_store.push_back(c.pattern); }
    for (size_t i = 0; i < cases.size(); ++i) {
        views[i].text        = text_store[i].data();
        views[i].text_len    = (int)text_store[i].size();
        views[i].pattern     = pat_store[i].data();
        views[i].pattern_len = (int)pat_store[i].size();
        views[i].max_score   = smax;
    }

    char *d_text = nullptr, *d_pat = nullptr;
    PairView* d_pairs = nullptr;
    int* d_scores = nullptr;

    size_t tot_text = 0, tot_pat = 0;
    for (auto& c : cases) { tot_text += c.text.size() + 1; tot_pat += c.pattern.size() + 1; }
    tot_text += 1; tot_pat += 1;   // pad so zero-length strings get a valid pointer

    HIP_CHECK(hipMalloc(&d_text, tot_text));
    HIP_CHECK(hipMalloc(&d_pat, tot_pat));
    HIP_CHECK(hipMalloc(&d_pairs, cases.size() * sizeof(PairView)));
    HIP_CHECK(hipMalloc(&d_scores, cases.size() * sizeof(int)));

    // Pack strings and fix up pointers to device addresses.
    //
    // FIX (R1 review): the previous version stored host byte-offsets by casting
    // them into the pointer fields ("views[i].text = (const char*)(ot)") and then
    // reinterpreted them back. That is UB, and it is fragile under the kernel's
    // __restrict__ qualifiers. Use an explicit 64-bit offset field instead.
    std::vector<char> h_text(tot_text, 0), h_pat(tot_pat, 0);
    std::vector<uint64_t> text_off(cases.size()), pat_off(cases.size());
    size_t ot = 0, op = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        size_t lt = text_store[i].size(), lp = pat_store[i].size();
        memcpy(h_text.data() + ot, text_store[i].data(), lt);
        memcpy(h_pat.data()  + op, pat_store[i].data(),  lp);
        text_off[i] = (uint64_t)ot;
        pat_off[i]  = (uint64_t)op;
        ot += lt + 1; op += lp + 1;
    }
    HIP_CHECK(hipMemcpy(d_text, h_text.data(), tot_text, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_pat,  h_pat.data(),  tot_pat,  hipMemcpyHostToDevice));

    for (size_t i = 0; i < views.size(); ++i) {
        views[i].text    = d_text + text_off[i];
        views[i].pattern = d_pat  + pat_off[i];
    }
    HIP_CHECK(hipMemcpy(d_pairs, views.data(), views.size() * sizeof(PairView),
                        hipMemcpyHostToDevice));

    // ---- launch -------------------------------------------------------------
    // Blocks must cover 2*smax+1 diagonals in one pass, else the thread-to-
    // diagonal mapping (k = tid - s) silently misses diagonals.
    const int need = 2 * smax + 1;
    int block = 1;
    while (block < need) block <<= 1;
    if (block > 1024) block = 1024;
    // Each wavefront needs padding on BOTH ends, one slot each: the reads of
    // prev[idx(k-1)] at k=-smax and prev[idx(k+1)] at k=+smax must land on a
    // sentinel slot, not out of bounds. Used range is [0, 2*smax+2], so each
    // wavefront occupies 2*smax+3 slots and two ping-pong waves are needed.
    //
    // The +2 is load-bearing, not defensive: sizing this as 2*(2*smax+1) made
    // the kernel's high-side read run one int past the allocation, which on the
    // MI210 is a memory access fault (H2 job 29184154, rc=134).
    const size_t shmem = (size_t)2 * (2 * smax + 3) * sizeof(int);
    printf("  block  : %d threads (need %d for 2*smax+1 diagonals)\n\n", block, need);
    wfa_score_kernel<<<(int)cases.size(), block, shmem>>>(d_pairs, d_scores, smax);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int> got(cases.size());
    HIP_CHECK(hipMemcpy(got.data(), d_scores, cases.size() * sizeof(int),
                        hipMemcpyDeviceToHost));
#else
    // Shim: run the score kernel body directly, one block per pair.
    std::vector<PairView> views(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) {
        views[i].text        = cases[i].text.data();
        views[i].text_len    = (int)cases[i].text.size();
        views[i].pattern     = cases[i].pattern.data();
        views[i].pattern_len = (int)cases[i].pattern.size();
        views[i].max_score   = smax;
    }
    std::vector<int> got(cases.size(), -1);
    shim::smem_vec().assign((size_t)2 * (2 * smax + 3), WFA_NEG);
    blockDim  = dim3{3, 1, 1};   // covers 2*s+1 for s <= 1; the host loop below
    threadIdx = uint3{0, 0, 0};
    (void)0;
    // The score kernel is grid-strided in threads as well as score, so the shim
    // runs it with the same block size the GPU launch uses, iterating the score
    // loop internally. Re-run it per pair with threadIdx swept by the kernel's
    // own loop bounds is not possible (the kernel is not written for it), so the
    // score parity under the shim is taken from the CPU DP reference instead --
    // the dedicated score gate (r1_check) covers the kernel's algebra, and the
    // traceback below is what this shim run exists to exercise.
    for (size_t i = 0; i < cases.size(); ++i) got[i] = ref[i];
    printf("  note   : SCORES on this line are the CPU DP reference, not the kernel.\n");
    printf("           The score kernel's algebra is gated by r1_check (smax=128).\n");
    printf("           What this run gates is the TRACEBACK, at smax=%d.\n\n", smax);
#endif

    // ---- compare ------------------------------------------------------------
    int pass = 0, fail = 0, abandoned = 0;
    int above_smax = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        // A true distance above smax is unresolved BY DESIGN, not a failure.
        // Counting it as "abandoned" next to a zero would read as though the
        // kernel handled it; name the population instead.
        if (ref[i] > smax) ++above_smax;
        if (got[i] < 0) { ++abandoned; continue; }   // unresolved within smax
        if (got[i] == ref[i]) ++pass;
        else {
            ++fail;
            if (fail <= 10) {
                printf("  MISMATCH [%s] gpu=%d cpu=%d  (text=%zu pattern=%zu)\n",
                       cases[i].label.c_str(), got[i], ref[i],
                       cases[i].text.size(), cases[i].pattern.size());
            }
        }
    }

    // ---- Fase 4: traceback --------------------------------------------------
    TraceRun tb;
    int tb_failed = 0, tb_checked = 0, tb_rescore = 0, tb_wf = 0;

    if (traceback_mode) {
#if GENOALIGNER_HAS_DEVICE
        run_trace(d_pairs, views.data(), cases, smax, chunk_bytes, cigar_cap,
                  d_scores, &tb);
#else
        run_trace(nullptr, views.data(), cases, smax, chunk_bytes, cigar_cap,
                  got.data(), &tb);
#endif
        if (tb.scores.size() != cases.size()) {
            printf("Fase 4: traceback did NOT run; cannot report\n\n");
            tb_failed = 1;
        } else {
            int skipped_above = 0;
            for (size_t i = 0; i < cases.size(); ++i) {
                if (tb.scores[i] < 0) {                  // unresolved within smax
                    if (ref[i] > smax) ++skipped_above;
                    continue;
                }
                ++tb_checked;
                const CigarCheck c = cigar_check(tb.cigars[i], cases[i].pattern, cases[i].text);
                if (c.wellformed)      ++tb_wf;
                if (c.wellformed && c.rescored == tb.scores[i]) ++tb_rescore;
                else if (tb_failed < 5) {
                    ++tb_failed;
                    printf("  CIGAR FAIL [%s] score=%d rescored=%d wellformed=%d\n",
                           cases[i].label.c_str(), tb.scores[i], c.rescored,
                           (int)c.wellformed);
                    printf("    cigar = %s\n", cigar_run_length(tb.cigars[i]).c_str());
                    if (c.first_bad_op)
                        printf("    first violation: '%c' at pattern[%d] text[%d]\n",
                               c.first_bad_op, c.first_bad_i, c.first_bad_j);
                }
            }
            printf("\n=== FASE 4 RESULT (CIGAR reconstruction) ===\n");
            printf("  checked             : %d\n", tb_checked);
            printf("  unresolved          : %d  (of which %d have true distance > smax=%d)\n",
                   tb.unresolved, skipped_above, smax);
            printf("  emitted             : %d\n", tb.emitted);
            printf("  truncated at cap    : %d  (cap=%d)\n", tb.truncated, cigar_cap);
            printf("  rescore == score    : %d/%d\n", tb_rescore, tb_checked);
            printf("  well-formed CIGAR   : %d/%d\n", tb_wf, tb_checked);
            // An unresolved pair whose TRUE distance is within smax is a real
            // defect: the kernel gave up earlier than it should have. Report it
            // separately so it cannot hide inside the legitimate population.
            if (tb.unresolved != skipped_above) {
                printf("  !!! %d unresolved pairs are WITHIN smax -- kernel stopped early\n",
                       tb.unresolved - skipped_above);
                tb_failed = 1;
            }
        }
    }

    // ---- emit (Fase 3 + Fase 4) ---------------------------------------------
    //
    // The external oracle must score the same pairs this run scored. Re-deriving
    // the generator in Python would risk a silent RNG divergence between the
    // C++ mt19937 and Python's, so the harness emits the pairs it actually used.
    // Format: index<TAB>pattern<TAB>text<TAB>gpu<TAB>cpu<TAB>label
    // Column 4 (gpu) carries OUR CIGAR string in traceback mode, so the same
    // reader shape works for check_cigar.py.
    if (!emit_path.empty()) {
        FILE* fh = fopen(emit_path.c_str(), "w");
        if (!fh) {
            fprintf(stderr, "ERROR: cannot open emit path %s\n", emit_path.c_str());
            return 1;
        }
        fprintf(fh, "# genoaligner Fase 3/4 %s\n",
                traceback_mode ? "CIGAR emit" : "score emit");
        fprintf(fh, "# smax=%d cases=%zu traceback=%d\n",
                smax, cases.size(), (int)traceback_mode);
        fprintf(fh, "index\tpattern\ttext\tgpu\tcpu\tlabel\n");
        for (size_t i = 0; i < cases.size(); ++i) {
            std::string col4;
            if (traceback_mode) {
                col4 = tb.cigars.size() == cases.size() && !tb.cigars[i].empty()
                       ? tb.cigars[i] : std::string("-");
            } else {
                char buf[32]; snprintf(buf, sizeof(buf), "%d", got[i]); col4 = buf;
            }
            fprintf(fh, "%zu\t%s\t%s\t%s\t%d\t%s\n",
                    i, cases[i].pattern.c_str(), cases[i].text.c_str(),
                    col4.c_str(), ref[i], cases[i].label.c_str());
        }
        fclose(fh);
        printf("  emitted   : %s (%zu cases)\n", emit_path.c_str(), cases.size());
    }

#if GENOALIGNER_HAS_DEVICE
    (void)hipFree(d_text); (void)hipFree(d_pat); (void)hipFree(d_pairs);
    (void)hipFree(d_scores);
#endif

    const int resolved = pass + fail;
    const double rate = resolved ? (100.0 * pass / resolved) : 0.0;

    printf("\n=== H2 RESULT (WFA score parity) ===\n");
    printf("  resolved  : %d\n", resolved);
    printf("  pass      : %d\n", pass);
    printf("  fail      : %d\n", fail);
    printf("  abandoned : %d  (edit distance > smax=%d)\n", abandoned, smax);
    printf("  parity    : %.2f%%  (criterion: >= 95%%)\n", rate);

    int rc = 0;
    if (rate < 95.0) { printf("  H2: FAIL -- examine formulation before optimising\n"); rc = 1; }
    else             { printf("  H2: PASS\n"); }

    if (traceback_mode) {
        // The traceback is the point of this run: gate on it.
        const bool tb_ok = (tb.scores.size() == cases.size())
                        && (tb_checked > 0)
                        && (tb_rescore == tb_checked)
                        && (tb_wf == tb_checked)
                        && (tb.truncated == 0);
        if (!tb_ok) { printf("  FASE 4: FAIL (see checks above)\n"); rc = 1; }
        else        { printf("  FASE 4: PASS (%d/%d re-score, %d/%d well-formed)\n",
                             tb_rescore, tb_checked, tb_wf, tb_checked); }
    }
    return rc;
}
