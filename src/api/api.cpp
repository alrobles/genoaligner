// genoaligner — public API implementation.
//
// Kept deliberately thin: it owns the launch, the buffers and the result
// validation, and nothing else. The kernels are in
// include/genoaligner/backend/wfa_kernel.hip, unchanged.
//
// TWO PATHS, ONE SOURCE
// ---------------------
// With GENOALIGNER_HIP_SHIM defined this builds and RUNS on CPU (the shim executes
// the real kernel body). Without it, it needs a GPU. That is what lets
// tests/api/test_api.cpp verify the API -- including the documented example -- on a
// machine with no accelerator, which is the same discipline the parity gate uses.
//
// The score kernel is the flat (fixed-block) one from Fase 7 -- merged as the
// default for the API's score-only path after being measured 1.5-4.4x faster
// AND free of the original kernel's 4.4x intra-process instability (idle lanes
// synchronising; docs/RESULTADO_FASE7_OPTIMIZACION.md). It uses the identical
// algebra, layout, sentinels and shared-memory envelope as the original, which
// remains in wfa_kernel.hip as the parity baseline the H2 gate checks.

// Include the public headers by the path a CONSUMER would use, resolved relative to
// this file. The previous form ("include/genoaligner/api.hpp", root-relative) worked
// in the build tree because the repository root was on the include path, and failed
// for every consumer that compiled against an installed tree with -I<prefix>/include.
// A library must include its own public headers the way it expects others to.
//
// HIP_ENABLE_WARP_SYNC_BUILTINS must be defined before the FIRST inclusion of
// hip_runtime.h in this TU (ROCm >= 5.7 hides the *_sync warp builtins behind
// it -- see sw_kernel.hip). wfa_kernel.hip below pulls hip_runtime.h in before
// sw_kernel_impl.hip could set it, so it is set here, above everything, rather
// than relying on include order surviving the next edit.
#if defined(__HIPCC__) && !defined(HIP_ENABLE_WARP_SYNC_BUILTINS)
#  define HIP_ENABLE_WARP_SYNC_BUILTINS
#endif
#include "genoaligner/api.hpp"
#include "genoaligner/backend/wfa_kernel.hip"
#include "genoaligner/backend/wfa_score_flat.hip"
#include "genoaligner/backend/sw_kernel_impl.hip"   // pulls in sw_kernel.hip

#include <cstdio>
#include <cstring>

#ifdef GENOALIGNER_HIP_SHIM
// The shim declares the launch-config globals extern and someone must define them.
// On the GPU paths the runtime provides them; under the shim this translation unit
// is the "launch owner", so it defines them -- required for the link to succeed even
// though the shim branch below never launches. Same arrangement as
// tests/parity/wfa_parity.cpp.
thread_local uint3 threadIdx{0, 0, 0};
uint3 blockIdx{0, 0, 0};
dim3  blockDim{1, 1, 1};
dim3  gridDim{1, 1, 1};
#endif

namespace genoaligner {

// ---- op codes -> chars. Kept local so the public API does not depend on the test
// tree; the codes are fixed by the kernel (0=M 1=X 2=I 3=D) and asserted in the
// API test against a known-good alignment, so a drift here fails loudly.
static char op_to_char(int op)
{
    switch (op) {
        case 0: return 'M';
        case 1: return 'X';
        case 2: return 'I';
        case 3: return 'D';
        default: return '?';
    }
}

static std::string decode_cigar(const int* rev, int used)
{
    std::string out;
    if (used <= 0) return out;
    out.resize((size_t)used);
    for (int j = 0; j < used; ++j) out[(size_t)j] = op_to_char(rev[used - 1 - j]);
    return out;
}

// Same three checks the Fase 4 gate applies, so anything leaving the API has passed
// them. repattern/text are needed: well-formedness is not a count test -- three
// development versions of the walk consumed the right counts while emitting M over
// mismatching positions.
static void validate(const std::string& cigar, int score,
                     const char* pattern, int m, const char* text, int n,
                     bool* rescore_ok, bool* wellformed_ok)
{
    int s = 0, pi = 0, ti = 0;
    bool wf = true;
    for (char c : cigar) {
        switch (c) {
            case 'M':
                if (pi >= m || ti >= n || pattern[pi] != text[ti]) wf = false;
                ++pi; ++ti; break;
            case 'X':
                if (pi >= m || ti >= n || pattern[pi] == text[ti]) wf = false;
                ++pi; ++ti; ++s; break;
            case 'I': ++ti; ++s; break;
            case 'D': ++pi; ++s; break;
            default:  wf = false; break;
        }
    }
    if (pi != m || ti != n) wf = false;      // must consume exactly both
    *wellformed_ok = wf;
    *rescore_ok    = (s == score);
}

// The SW twin of validate(): re-score a local-alignment CIGAR under the affine
// scheme over the ALIGNED span [start_i, end_i] x [start_j, end_j]. A run of k
// consecutive I's or D's is ONE gap of length k costing go + (k-1)*ge -- the
// recurrence's own convention. text_len/pattern_len guards are kept alongside
// the span bounds: a kernel bug that reported a coordinate past the sequence
// end must be caught, not read past.
static void sw_validate(const std::string& cigar,
                        const char* text, int text_len, int start_i, int end_i,
                        const char* pattern, int pattern_len, int start_j, int end_j,
                        int match, int mismatch, int gap_open, int gap_extend,
                        int score, bool* rescore_ok, bool* wellformed_ok)
{
    int i = start_i, j = start_j, s = 0, run = 0;   // run: +k in I, -k in D
    bool wf = true;
    auto close_gap = [&] {
        if (run != 0) {
            const int len = (run > 0) ? run : -run;
            s -= gap_open + (len - 1) * gap_extend;
            run = 0;
        }
    };
    for (char c : cigar) {
        switch (c) {
            case 'M':
                close_gap();
                if (i > end_i || j > end_j || i >= text_len || j >= pattern_len ||
                    text[i] != pattern[j]) wf = false;
                ++i; ++j; s += match; break;
            case 'X':
                close_gap();
                if (i > end_i || j > end_j || i >= text_len || j >= pattern_len ||
                    text[i] == pattern[j]) wf = false;
                ++i; ++j; s += mismatch; break;
            case 'I':
                if (run < 0) close_gap();
                if (i > end_i || i >= text_len) wf = false;
                ++i; ++run; break;
            case 'D':
                if (run > 0) close_gap();
                if (j > end_j || j >= pattern_len) wf = false;
                ++j; --run; break;
            default: wf = false; break;
        }
    }
    close_gap();
    if (i != end_i + 1 || j != end_j + 1) wf = false;   // must land exactly
    *wellformed_ok = wf;
    *rescore_ok    = wf && (s == score);
}

// ---------------------------------------------------------------------------
// Device helpers. Under the shim there is no device, and the shim path below
// never calls these, but they must still exist so callers can query the API
// uniformly.
// ---------------------------------------------------------------------------
#ifdef GENOALIGNER_HIP_SHIM
const char* backend_name() { return "cpu-shim"; }
const char* device_name()  { return "none (CPU shim)"; }
bool        device_available() { return false; }
#else
#ifdef __HIP_PLATFORM_NVIDIA__
static const char* g_backend = "cuda";
#else
static const char* g_backend = "rocm";
#endif
const char* backend_name() { return g_backend; }
bool device_available()
{
    int n = 0;
    return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}
const char* device_name()
{
    // Thread-safe memoisation, because the first version was not: a bare
    // `static char buf[256]` with `if (buf[0]) return buf;` is a check-then-write
    // data race -- two threads can both see an empty buffer and both snprintf into
    // it. The effect happened to be benign (both write the same string) but "benign
    // race" is not a property to rely on, and a caller has no way to know.
    //
    // Function-local static initialisation in C++11+ is guaranteed thread-safe, so
    // this fixes it with no lock and no ordering assumptions.
    static const std::string name = [] {
        hipDeviceProp_t p{};
        if (hipGetDeviceProperties(&p, 0) != hipSuccess) return std::string("none");
        return std::string(p.name);
    }();
    return name.c_str();
}
#endif

// ---------------------------------------------------------------------------
// The one place buffers are sized. Kept as a small struct so the single-pair and
// batch paths cannot drift in their sizing -- the Fase 6 bug where the bench passed
// 0 bytes of shared memory came from the launch site re-deriving size arithmetic.
// ---------------------------------------------------------------------------
namespace {

struct LaunchCfg {
    int    block;     // threads per block
    size_t shmem;     // dynamic shared bytes
};

static LaunchCfg cfg_for(int smax)
{
    LaunchCfg c{};
    // The flat kernel sweeps diagonals in an interior grid-stride loop, so its
    // block does NOT need to cover 2*smax+1 slots -- WFA_FLAT_BLOCK is a cost
    // choice, not a coverage choice, and smaller smax requests do not shrink it
    // (see wfa_score_flat.hip for why a fixed block was the whole point).
    c.block  = WFA_FLAT_BLOCK;
    c.shmem  = (size_t)2 * (2 * smax + 3) * sizeof(int);
    return c;
}

// The API contract caps smax at 511. The original kernel NEEDED that cap (its
// thread-to-diagonal mapping could not cover [0,2s] past a 1024-thread block and
// would silently skip diagonals). The flat kernel's grid-stride has no such
// limit -- only the shared-memory envelope -- but the contract stays at the
// validated bound rather than silently widening it.
static constexpr int kMaxSmax = 511;

}  // namespace

// ---------------------------------------------------------------------------
// Batch: all the device work lives here; align() is a one-element batch.
// ---------------------------------------------------------------------------
BatchResult align_batch(const std::vector<AlignRequest>& reqs)
{
    BatchResult out;
    out.results.resize(reqs.size());
    if (reqs.empty()) return out;

    const int N = (int)reqs.size();

    // ---- input validation. Refuse; do not degrade silently. --------------
    // Each of these would otherwise produce a plausible-looking wrong answer, which
    // is the one outcome this library must never produce.
    out.results.assign((size_t)N, AlignResult{});
    for (int i = 0; i < N; ++i) {
        const AlignRequest& r = reqs[(size_t)i];
        if ((r.pattern_len > 0 && r.pattern == nullptr) ||
            (r.text_len    > 0 && r.text    == nullptr)) {
            out.status = BatchResult::Status::invalid_argument;
            out.error  = "null pointer with non-zero length";
            return out;
        }
        if (r.pattern_len < 0 || r.text_len < 0) {
            out.status = BatchResult::Status::invalid_argument;
            out.error  = "negative length";
            return out;
        }
        if (r.smax < 0 || r.smax > kMaxSmax) {
            // See kMaxSmax: above this the thread-to-diagonal mapping skips diagonals
            // silently, so an out-of-range smax yields WRONG ALIGNMENTS, not an error.
            out.status = BatchResult::Status::invalid_argument;
            out.error  = "smax out of range [0, 511]";
            return out;
        }
    }

    // One smax for the whole batch. The kernels take a single smax, and mixing them
    // in one launch would silently use the wrong bound for some pairs, so the batch
    // takes the minimum and SAYS SO rather than pretending to be per-pair.
    int smax = reqs[0].smax;
    for (const auto& r : reqs) if (r.smax < smax) smax = r.smax;
    bool any_cigar = false;
    for (const auto& r : reqs) if (r.with_cigar) any_cigar = true;

    // Pack strings into stable storage.
    std::string P, T;
    std::vector<int> poff(N), toff(N);
    for (int i = 0; i < N; ++i) {
        poff[i] = (int)P.size(); P += std::string(reqs[i].pattern, (size_t)reqs[i].pattern_len);
        toff[i] = (int)T.size(); T += std::string(reqs[i].text,    (size_t)reqs[i].text_len);
    }

#ifdef GENOALIGNER_HIP_SHIM
    // CPU-shim path. The shim CANNOT score through the kernel: with blockDim=1 the
    // score kernel's `threadIdx.x <= 2*s` guard evaluates for a single diagonal, so
    // the shim score is degenerate (documented in bench/h7_flat_parity.cpp, where it
    // resolved 138 of 1230 cases). Rather than fake a kernel result, this path
    // declares itself unavailable and the API test drives the real kernel through
    // the parity harness instead.
    //
    // What the shim path DOES verify is the API's shape: that requests pack and
    // unpack, that results come back in order, and that the counters add up.
    (void)P; (void)T; (void)poff; (void)toff;
    out.unresolved_count = N;
    for (int i = 0; i < N; ++i) {
        out.results[(size_t)i].score    = -1;
        out.results[(size_t)i].resolved = false;
    }
    return out;
#else
    // ---- real device path ----
    PairView* d_pairs = nullptr;
    char*     d_pat   = nullptr;
    char*     d_tex   = nullptr;
    int*      d_scores = nullptr;
    int*      d_cg     = nullptr;
    int*      d_meta   = nullptr;

    // CIGAR capacity must clear the WHOLE batch, not the first request. Sizing it
    // from reqs[0] (the version that shipped) truncated every pair longer than the
    // first one, and the truncation surfaced as an empty CIGAR -- which then failed
    // validation for 190 of 200 pairs while the SCORES were perfect. A value derived
    // from one element and applied to a collection is the same bug class this
    // project keeps finding; here the test caught it on the first GPU run.
    int cigar_cap = 1;
    for (const auto& r : reqs) {
        const int need = r.pattern_len + r.text_len + 2;
        if (need > cigar_cap) cigar_cap = need;
    }
    auto free_all = [&]() {
        if (d_pairs) hipFree(d_pairs);
        if (d_pat)   hipFree(d_pat);
        if (d_tex)   hipFree(d_tex);
        if (d_scores) hipFree(d_scores);
        if (d_cg)    hipFree(d_cg);
        if (d_meta)  hipFree(d_meta);
    };
    // One place that turns a device failure into a REPORTED failure rather than a
    // batch full of -1 that reads like a legitimate result.
    auto fail = [&](const char* what) -> BatchResult {
        free_all();
        out.status = BatchResult::Status::device_error;
        out.error  = what;
        out.results.assign((size_t)N, AlignResult{});
        out.resolved_count = 0;
        out.unresolved_count = 0;
        return out;
    };

    if (hipMalloc(&d_pairs, (size_t)N * sizeof(PairView)) != hipSuccess) return fail("hipMalloc(d_pairs)");
    if (hipMalloc(&d_pat, P.size() ? P.size() : 1) != hipSuccess)        return fail("hipMalloc(d_pat)");
    if (hipMalloc(&d_tex, T.size() ? T.size() : 1) != hipSuccess)        return fail("hipMalloc(d_tex)");
    if (hipMalloc(&d_scores, (size_t)N * sizeof(int)) != hipSuccess)     return fail("hipMalloc(d_scores)");
    // Sentinel before any kernel runs: a score buffer that stays -1 means "the
    // kernel never wrote it". Without this, a dead kernel leaves uninitialised
    // device memory that memcpy's back as scores that look VALID -- the one
    // output this library must never produce.
    if (hipMemset(d_scores, 0xFF, (size_t)N * sizeof(int)) != hipSuccess)
        return fail("hipMemset(d_scores)");
    if (any_cigar) {
        if (hipMalloc(&d_cg, (size_t)N * (size_t)cigar_cap * sizeof(int)) != hipSuccess) return fail("hipMalloc(d_cg)");
        if (hipMalloc(&d_meta, (size_t)N * 4 * sizeof(int)) != hipSuccess) return fail("hipMalloc(d_meta)");
        if (hipMemset(d_meta, 0, (size_t)N * 4 * sizeof(int)) != hipSuccess) return fail("hipMemset(d_meta)");
    }

    hipError_t cpy;
    cpy = hipMemcpy(d_pat, P.data(), P.size(), hipMemcpyHostToDevice);
    if (cpy != hipSuccess) return fail("hipMemcpy(pattern)");
    cpy = hipMemcpy(d_tex, T.data(), T.size(), hipMemcpyHostToDevice);
    if (cpy != hipSuccess) return fail("hipMemcpy(text)");

    // Drain any error latched by the setup above. Without this, a failure that
    // happened BEFORE the launch is reported by the hipGetLastError() after it, and
    // the error message blames the kernel for something it did not do. That is
    // exactly what made "wfa_trace_kernel launch" appear on a call whose real
    // problem was elsewhere.
    (void)hipGetLastError();

    // Views carry DEVICE pointers -- built after the string copy, never memcpy'd
    // from host-built structs (that bug is documented in bench/tcus.cpp).
    std::vector<PairView> views((size_t)N);
    for (int i = 0; i < N; ++i) {
        views[(size_t)i] = PairView{ d_tex + toff[i], reqs[i].text_len,
                                     d_pat + poff[i], reqs[i].pattern_len, smax };
    }
    cpy = hipMemcpy(d_pairs, views.data(), (size_t)N * sizeof(PairView), hipMemcpyHostToDevice);
    if (cpy != hipSuccess) return fail("hipMemcpy(views)");

    if (any_cigar) {
        // Traceback path: score + CIGAR in one kernel.
        //
        // chunk_bytes is the 4th kernel argument AND the dynamic shared-memory size
        // of the launch. The first version of this API passed cfg_for()'s shmem here,
        // which is the SCORE kernel's size -- a different quantity -- and every
        // traceback launch failed with "wfa_trace_kernel launch". The value must
        // match what the kernel expects (the harness, which works, computes it this
        // way), and if it exceeds the 48 KB default the attribute has to be raised
        // BEFORE the launch or the driver rejects it.
        const int wf_stride    = 2 * smax + 3;
        const int wf_alloc_max = (smax + 1) * wf_stride;
        const int chunk_bytes  = wf_alloc_max * (int)sizeof(int);

        int* d_ws = nullptr;
        if (hipMalloc(&d_ws, (size_t)wf_alloc_max * (size_t)N * sizeof(int)) != hipSuccess) {
            return fail("hipMalloc(d_ws)");
        }
        if (hipMemset(d_ws, 0, (size_t)wf_alloc_max * (size_t)N * sizeof(int)) != hipSuccess) {
            hipFree(d_ws);
            return fail("hipMemset(d_ws)");
        }

        if (chunk_bytes > 48 * 1024) {
            hipError_t as = hipFuncSetAttribute(
                (const void*)wfa_trace_kernel,
                hipFuncAttributeMaxDynamicSharedMemorySize, chunk_bytes);
            if (as != hipSuccess) {
                hipFree(d_ws);
                // Ask the device for the limit that DOES exist in ROCm 6.4.3
                // (MaxSharedMemoryPerBlockOptin is CUDA-only there -- referencing it
                // did not compile, which is how this branch was found to be broken
                // in the first place) and report it, so the message is actionable
                // instead of "invalid argument".
                int max_smem = 0;
                hipDeviceGetAttribute(&max_smem, hipDeviceAttributeMaxSharedMemoryPerBlock, 0);
                std::fprintf(stderr,
                             "[genoaligner] smem request %d B exceeds what the device "
                             "accepts (%d B per block). Lower smax, or pass "
                             "with_cigar=false to score only.\n",
                             chunk_bytes, max_smem);
                return fail("shared memory for this smax exceeds the device limit");
            }
        }

        hipLaunchKernelGGL(wfa_trace_kernel, dim3((unsigned)N), dim3(1),
                           (size_t)chunk_bytes,
                           (hipStream_t)0, d_pairs, d_scores, d_cg, d_meta, d_ws,
                           smax, wf_stride, wf_alloc_max, chunk_bytes, cigar_cap);
        hipError_t le = hipGetLastError();
        if (le != hipSuccess) {
            hipFree(d_ws);
            // Report WHICH error, not just that there was one. The first version said
            // "launch failed" for every cause and cost a debugging cycle.
            std::fprintf(stderr,
                         "[genoaligner] trace launch failed: %s (block=%u, smem=%d B, "
                         "cigar_cap=%d, wf_alloc_max=%d, N=%u)\n",
                         hipGetErrorString(le), 1u, chunk_bytes, cigar_cap, wf_alloc_max,
                         (unsigned)N);
            return fail(hipGetErrorString(le));
        }
        // The launch returning no error only means the kernel STARTED. A fault
        // during execution surfaces here -- and unchecked, it would leave the
        // -1 sentinel / device garbage in d_scores to be read as real results.
        hipError_t se = hipDeviceSynchronize();
        if (se != hipSuccess) { hipFree(d_ws); return fail(hipGetErrorString(se)); }

        std::vector<int> scores((size_t)N, -1), meta((size_t)N * 4, 0);
        std::vector<int> cg((size_t)N * (size_t)cigar_cap, 0);
        if (hipMemcpy(scores.data(), d_scores, (size_t)N * sizeof(int), hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(meta.data(),   d_meta,   (size_t)N * 4 * sizeof(int), hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(cg.data(),     d_cg,     (size_t)N * (size_t)cigar_cap * sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
            hipFree(d_ws);
            return fail("hipMemcpy(results)");
        }

        for (int i = 0; i < N; ++i) {
            const int rev_used = meta[(size_t)i * 4 + 1];
            const int trunc    = meta[(size_t)i * 4 + 2];
            AlignResult& r = out.results[(size_t)i];
            r.score    = scores[(size_t)i];
            r.resolved = r.score >= 0;
            if (r.resolved && !trunc && rev_used > 0) {
                r.cigar = decode_cigar(cg.data() + (size_t)i * (size_t)cigar_cap, rev_used);
                validate(r.cigar, r.score,
                         reqs[i].pattern, reqs[i].pattern_len,
                         reqs[i].text,    reqs[i].text_len,
                         &r.rescore_ok, &r.wellformed_ok);
            }
        }
        hipFree(d_ws);
    } else {
        // Score-only path: always the flat kernel (merged default -- see the
        // header comment for the evidence and why the original stays in
        // wfa_kernel.hip as the parity baseline).
        LaunchCfg c = cfg_for(smax);
        hipLaunchKernelGGL(wfa_score_kernel_flat, dim3((unsigned)N), dim3(c.block), c.shmem,
                           (hipStream_t)0, d_pairs, d_scores, smax);
        hipError_t le = hipGetLastError();
        if (le != hipSuccess) { return fail("wfa_score_kernel launch"); }
        hipError_t se = hipDeviceSynchronize();
        if (se != hipSuccess) return fail(hipGetErrorString(se));
        std::vector<int> scores((size_t)N, -1);
        if (hipMemcpy(scores.data(), d_scores, (size_t)N * sizeof(int), hipMemcpyDeviceToHost) != hipSuccess)
            return fail("hipMemcpy(scores)");
        for (int i = 0; i < N; ++i) {
            out.results[(size_t)i].score    = scores[(size_t)i];
            out.results[(size_t)i].resolved = scores[(size_t)i] >= 0;
        }
    }
    free_all();
#endif

    for (const auto& r : out.results) {
        if (r.resolved) ++out.resolved_count; else ++out.unresolved_count;
    }
    return out;
}

AlignResult align(const AlignRequest& req)
{
    std::vector<AlignRequest> one{req};
    BatchResult b = align_batch(one);
    return b.results.empty() ? AlignResult{} : b.results[0];
}

// ===========================================================================
// SMITH-WATERMAN — api.hpp has the contract; this is the plumbing.
// ===========================================================================
namespace {

// All the rules in api.hpp's "scoring scheme" section, in one place so the
// single-pair and batch paths cannot drift. `with_cigar` matters: gap_extend >
// gap_open is degenerate only for the CIGAR, so score-only requests still take
// it -- the score stays exact under that regime.
static bool sw_scheme_ok(const SWScoring& s, bool with_cigar, const char** err)
{
    if (s.match <= 0) {
        *err = "SW scoring: match must be > 0 (the score is 0 for every input otherwise)";
        return false;
    }
    if (s.mismatch >= s.match) {
        *err = "SW scoring: mismatch must be < match";
        return false;
    }
    if (s.gap_open <= 0 || s.gap_extend <= 0) {
        *err = "SW scoring: gap costs must be > 0";
        return false;
    }
    if (with_cigar && s.gap_extend > s.gap_open) {
        *err = "SW scoring: gap_extend > gap_open with with_cigar cannot re-score "
               "to its own score (api.hpp; RESULTADO_H9_SW2_TRACE.md). Score-only "
               "requests may use it";
        return false;
    }
    return true;
}

// Fill results from the trace kernel's outputs -- shared by the GPU path and
// the CPU-shim path so they cannot disagree on what a result means.
// SW_TRACE_TOO_BIG is a kernel-level sentinel; the API translates it to
// too_large rather than leaking it as a score.
static void sw_fill_trace_results(SWBatchResult& out,
                                  const std::vector<SWRequest>& reqs,
                                  const std::vector<SWTraceResult>& tres,
                                  const std::vector<int>& meta,
                                  const std::vector<int>& cg, int cigar_cap)
{
    const int N = (int)reqs.size();
    for (int i = 0; i < N; ++i) {
        const SWTraceResult& t = tres[(size_t)i];
        SWAlignResult& r = out.results[(size_t)i];
        if (t.score == SW_TRACE_TOO_BIG) {
            r.too_large = true;             // resolved stays false, score -1
            continue;
        }
        if (t.score < 0) continue;          // sentinel: kernel never wrote it
        r.score    = t.score;
        r.resolved = true;
        r.end_i    = t.end_i;
        r.end_j    = t.end_j;
        if (t.score == 0) {
            r.rescore_ok = r.wellformed_ok = true;   // empty alignment, vacuous
            continue;
        }
        if (!reqs[(size_t)i].with_cigar) continue;     // score+end only, per contract
        r.start_i = t.start_i;
        r.start_j = t.start_j;
        const int used  = meta[(size_t)i * 4 + 1];
        const int trunc = meta[(size_t)i * 4 + 2];
        if (!trunc && used > 0) {
            r.cigar = decode_cigar(cg.data() + (size_t)i * (size_t)cigar_cap, used);
            const SWScoring& s = reqs[(size_t)i].scoring;
            sw_validate(r.cigar,
                        reqs[i].text, reqs[i].text_len, r.start_i, r.end_i,
                        reqs[i].pattern, reqs[i].pattern_len, r.start_j, r.end_j,
                        s.match, s.mismatch, s.gap_open, s.gap_extend,
                        r.score, &r.rescore_ok, &r.wellformed_ok);
        }
    }
}

}  // namespace

SWBatchResult align_sw_batch(const std::vector<SWRequest>& reqs)
{
    SWBatchResult out;
    out.results.resize(reqs.size());
    if (reqs.empty()) return out;

    const int N = (int)reqs.size();

    auto invalid = [&](const char* msg) -> SWBatchResult {
        out.status = SWBatchResult::Status::invalid_argument;
        out.error  = msg;
        out.results.assign((size_t)N, SWAlignResult{});
        return out;
    };

    // ---- input validation: refuse, never degrade. All pre-device. ---------
    for (int i = 0; i < N; ++i) {
        const SWRequest& r = reqs[(size_t)i];
        if ((r.text_len    > 0 && r.text    == nullptr) ||
            (r.pattern_len > 0 && r.pattern == nullptr))
            return invalid("null pointer with non-zero length");
        if (r.text_len < 0 || r.pattern_len < 0)
            return invalid("negative length");
        const char* e = nullptr;
        if (!sw_scheme_ok(r.scoring, r.with_cigar, &e)) return invalid(e);
        // One launch takes ONE scheme (same reason as the one-smax rule). A
        // batch that mixes schemes would silently use the first one's
        // parameters for every pair.
        const SWScoring& s0 = reqs[0].scoring;
        if (r.scoring.match != s0.match || r.scoring.mismatch != s0.mismatch ||
            r.scoring.gap_open != s0.gap_open || r.scoring.gap_extend != s0.gap_extend)
            return invalid("all requests in an SW batch must share one scoring scheme");
    }

    const SWParams prm{ reqs[0].scoring.match, reqs[0].scoring.mismatch,
                        reqs[0].scoring.gap_open, reqs[0].scoring.gap_extend };

    bool any_cigar = false;
    for (const auto& r : reqs) if (r.with_cigar) any_cigar = true;

    // Pack strings into stable storage. NOTE the field order: SWPairView is
    // {text, text_len, pattern, pattern_len} -- text first, unlike PairView.
    std::string T, P;
    std::vector<int> toff((size_t)N), poff((size_t)N);
    for (int i = 0; i < N; ++i) {
        toff[(size_t)i] = (int)T.size(); T += std::string(reqs[i].text,    (size_t)reqs[i].text_len);
        poff[(size_t)i] = (int)P.size(); P += std::string(reqs[i].pattern, (size_t)reqs[i].pattern_len);
    }

    // Workspace sizing, one place (the cigar_cap lesson: size from the batch
    // maximum, never the first request). dir_stride is capped at the declared
    // limit; a pair whose cells exceed it comes back too_large.
    size_t max_cells = 0;
    int max_n = 0, max_m = 0;
    for (const auto& r : reqs) {
        const size_t cells = ((size_t)r.text_len + 1) * ((size_t)r.pattern_len + 1);
        if (cells > max_cells) max_cells = cells;
        if (r.pattern_len > max_n) max_n = r.pattern_len;
        if (r.text_len    > max_m) max_m = r.text_len;
    }
    const size_t dir_stride     = max_cells < (size_t)SW_MAX_TRACE_CELLS
                                  ? max_cells : (size_t)SW_MAX_TRACE_CELLS;
    const size_t scratch_stride = 2 * ((size_t)max_n + 1);
    const int    cigar_cap      = max_m + max_n + 4;

#ifdef GENOALIGNER_HIP_SHIM
    // CPU-shim path -- and here SW differs fundamentally from WFA: the SW
    // kernels CAN run for real on host memory, so this path produces actual
    // results instead of declaring itself unavailable. with_cigar pairs run
    // the trace kernel one-thread-per-pair (direct call, as test_sw_trace does);
    // a pure score-only batch runs the real score kernel under the shim's warp
    // emulation, which is the same body the GPU executes.
    std::vector<SWPairView> pv((size_t)N);
    for (int i = 0; i < N; ++i) {
        pv[(size_t)i] = SWPairView{ T.data() + toff[(size_t)i], reqs[i].text_len,
                                    P.data() + poff[(size_t)i], reqs[i].pattern_len };
    }

    if (any_cigar) {
        std::vector<SWTraceResult> tres((size_t)N, SWTraceResult{-1, -1, -1, -1, -1});
        std::vector<int>           meta((size_t)N * 4, 0);
        std::vector<int>           cg((size_t)N * (size_t)cigar_cap, 0);
        std::vector<uint8_t>       dirs((size_t)N * dir_stride, 0);
        std::vector<int>           scratch((size_t)N * scratch_stride, 0);
        for (int i = 0; i < N; ++i) {
            blockIdx = uint3{(unsigned)i, 0, 0};
            sw_trace_kernel(pv.data(), prm, tres.data(), cg.data(), meta.data(),
                            dirs.data(), dir_stride, scratch.data(), scratch_stride,
                            N, cigar_cap);
        }
        blockIdx = uint3{0, 0, 0};
        sw_fill_trace_results(out, reqs, tres, meta, cg, cigar_cap);
    } else {
        std::vector<SWResult> sres((size_t)N, SWResult{-1, -1, -1});
        const int nthreads   = SW_WARPS_PER_BLOCK * warpSize;
        const int ints_per_w = sw_smem_ints_per_warp(max_n);
        shim::smem_vec().assign((size_t)SW_WARPS_PER_BLOCK * (size_t)ints_per_w, 0);
        blockDim = dim3{(unsigned)nthreads, 1, 1};
        const int nblocks = (N + SW_WARPS_PER_BLOCK - 1) / SW_WARPS_PER_BLOCK;
        for (int b = 0; b < nblocks; ++b) {
            blockIdx = uint3{(unsigned)b, 0, 0};
            shim::run_block(nthreads, [&] {
                sw_score_kernel(pv.data(), prm, sres.data(), N, ints_per_w);
            });
        }
        blockDim = dim3{1, 1, 1}; blockIdx = uint3{0, 0, 0};
        for (int i = 0; i < N; ++i) {
            const SWResult& t = sres[(size_t)i];
            SWAlignResult& r = out.results[(size_t)i];
            if (t.score < 0) continue;      // sentinel: kernel never wrote
            r.score    = t.score;
            r.resolved = true;
            r.end_i    = t.end_i;
            r.end_j    = t.end_j;
            if (t.score == 0) { r.rescore_ok = r.wellformed_ok = true; }
        }
    }
#else
    // ---- real device path ----
    SWPairView*    d_pairs = nullptr;
    char*          d_tex   = nullptr;
    char*          d_pat   = nullptr;
    SWTraceResult* d_tres  = nullptr;
    SWResult*      d_sres  = nullptr;
    int*           d_cg    = nullptr;
    int*           d_meta  = nullptr;
    uint8_t*       d_dirs  = nullptr;
    int*           d_scr   = nullptr;

    auto free_all = [&]() {
        if (d_pairs) hipFree(d_pairs);
        if (d_tex)   hipFree(d_tex);
        if (d_pat)   hipFree(d_pat);
        if (d_tres)  hipFree(d_tres);
        if (d_sres)  hipFree(d_sres);
        if (d_cg)    hipFree(d_cg);
        if (d_meta)  hipFree(d_meta);
        if (d_dirs)  hipFree(d_dirs);
        if (d_scr)   hipFree(d_scr);
    };
    auto fail = [&](const char* what) -> SWBatchResult {
        free_all();
        out.status = SWBatchResult::Status::device_error;
        out.error  = what;
        out.results.assign((size_t)N, SWAlignResult{});
        out.resolved_count = 0;
        out.unresolved_count = 0;
        return out;
    };

    if (hipMalloc(&d_pairs, (size_t)N * sizeof(SWPairView)) != hipSuccess)   return fail("hipMalloc(d_pairs)");
    if (hipMalloc(&d_tex, T.size() ? T.size() : 1) != hipSuccess)            return fail("hipMalloc(d_tex)");
    if (hipMalloc(&d_pat, P.size() ? P.size() : 1) != hipSuccess)            return fail("hipMalloc(d_pat)");
    if (hipMemcpy(d_tex, T.data(), T.size(), hipMemcpyHostToDevice) != hipSuccess) return fail("hipMemcpy(text)");
    if (hipMemcpy(d_pat, P.data(), P.size(), hipMemcpyHostToDevice) != hipSuccess) return fail("hipMemcpy(pattern)");

    // Views carry DEVICE pointers -- the SWPairView host-pointer bug is
    // documented in Fase B.
    std::vector<SWPairView> views((size_t)N);
    for (int i = 0; i < N; ++i) {
        views[(size_t)i] = SWPairView{ d_tex + toff[(size_t)i], reqs[i].text_len,
                                       d_pat + poff[(size_t)i], reqs[i].pattern_len };
    }
    if (hipMemcpy(d_pairs, views.data(), (size_t)N * sizeof(SWPairView),
                  hipMemcpyHostToDevice) != hipSuccess) return fail("hipMemcpy(views)");

    if (any_cigar) {
        // Traceback: one thread per pair over a direction-byte table.
        if (hipMalloc(&d_tres, (size_t)N * sizeof(SWTraceResult)) != hipSuccess)     return fail("hipMalloc(results)");
        if (hipMalloc(&d_meta, (size_t)N * 4 * sizeof(int)) != hipSuccess)           return fail("hipMalloc(meta)");
        if (hipMalloc(&d_cg, (size_t)N * (size_t)cigar_cap * sizeof(int)) != hipSuccess) return fail("hipMalloc(cigar)");
        if (hipMalloc(&d_dirs, (size_t)N * dir_stride) != hipSuccess)
            return fail("hipMalloc(dir table) -- batch workspace exceeds device memory; split the batch");
        if (hipMalloc(&d_scr, (size_t)N * scratch_stride * sizeof(int)) != hipSuccess)
            return fail("hipMalloc(scratch)");
        // Sentinel: an unwritten result must read as "not computed", never as a
        // plausible score. 0xFF fills every int field with -1.
        if (hipMemset(d_tres, 0xFF, (size_t)N * sizeof(SWTraceResult)) != hipSuccess)
            return fail("hipMemset(results)");
        if (hipMemset(d_meta, 0, (size_t)N * 4 * sizeof(int)) != hipSuccess)
            return fail("hipMemset(meta)");

        (void)hipGetLastError();   // drain setup errors before blaming the launch
        const int tblock = 64;
        const int tgrid  = (N + tblock - 1) / tblock;
        hipLaunchKernelGGL(sw_trace_kernel, dim3((unsigned)tgrid), dim3((unsigned)tblock),
                           0, (hipStream_t)0, d_pairs, prm, d_tres, d_cg, d_meta,
                           d_dirs, dir_stride, d_scr, scratch_stride, N, cigar_cap);
        hipError_t le = hipGetLastError();
        if (le != hipSuccess) return fail(hipGetErrorString(le));
        hipError_t se = hipDeviceSynchronize();
        if (se != hipSuccess) return fail(hipGetErrorString(se));

        std::vector<SWTraceResult> tres((size_t)N);
        std::vector<int> meta((size_t)N * 4, 0);
        std::vector<int> cg((size_t)N * (size_t)cigar_cap, 0);
        if (hipMemcpy(tres.data(), d_tres, (size_t)N * sizeof(SWTraceResult), hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(meta.data(), d_meta, (size_t)N * 4 * sizeof(int), hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(cg.data(),   d_cg,   (size_t)N * (size_t)cigar_cap * sizeof(int), hipMemcpyDeviceToHost) != hipSuccess)
            return fail("hipMemcpy(results)");
        sw_fill_trace_results(out, reqs, tres, meta, cg, cigar_cap);
    } else {
        // Score only: the warp-per-row kernel. Its pattern_len is bounded by
        // device shared memory -- a launch-level resource, so exceeding it is
        // refused for the whole batch, not degraded per pair.
        int max_smem = 0;
        hipDeviceGetAttribute(&max_smem, hipDeviceAttributeMaxSharedMemoryPerBlock, 0);
        const int    ints_per_w = sw_smem_ints_per_warp(max_n);
        const size_t smem       = (size_t)SW_WARPS_PER_BLOCK * (size_t)ints_per_w * sizeof(int);
        if (max_smem <= 0 || smem > (size_t)max_smem) {
            free_all();      // allocs already happened; invalid() does not free
            return invalid("pattern_len exceeds this device's shared-memory limit "
                           "for the score kernel (see api.hpp supported-size limits)");
        }
        if (hipMalloc(&d_sres, (size_t)N * sizeof(SWResult)) != hipSuccess)
            return fail("hipMalloc(results)");
        if (hipMemset(d_sres, 0xFF, (size_t)N * sizeof(SWResult)) != hipSuccess)
            return fail("hipMemset(results)");

        // Above the default 48 KiB the driver must opt in BEFORE the launch --
        // the WFA trace path learned this the hard way.
        if (smem > 48 * 1024) {
            hipError_t as = hipFuncSetAttribute(
                (const void*)sw_score_kernel,
                hipFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            if (as != hipSuccess)
                return fail("shared memory for this pattern_len exceeds the device limit");
        }

        int dev_warp = 0;
        hipDeviceGetAttribute(&dev_warp, hipDeviceAttributeWarpSize, 0);
        const int sblock = SW_WARPS_PER_BLOCK * dev_warp;
        const int sgrid  = (N + SW_WARPS_PER_BLOCK - 1) / SW_WARPS_PER_BLOCK;
        (void)hipGetLastError();
        hipLaunchKernelGGL(sw_score_kernel, dim3((unsigned)sgrid), dim3((unsigned)sblock),
                           smem, (hipStream_t)0, d_pairs, prm, d_sres, N, ints_per_w);
        hipError_t le = hipGetLastError();
        if (le != hipSuccess) return fail(hipGetErrorString(le));
        hipError_t se = hipDeviceSynchronize();
        if (se != hipSuccess) return fail(hipGetErrorString(se));

        std::vector<SWResult> sres((size_t)N);
        if (hipMemcpy(sres.data(), d_sres, (size_t)N * sizeof(SWResult),
                      hipMemcpyDeviceToHost) != hipSuccess)
            return fail("hipMemcpy(results)");
        for (int i = 0; i < N; ++i) {
            const SWResult& t = sres[(size_t)i];
            SWAlignResult& r = out.results[(size_t)i];
            if (t.score < 0) continue;      // sentinel: kernel never wrote
            r.score    = t.score;
            r.resolved = true;
            r.end_i    = t.end_i;
            r.end_j    = t.end_j;
            if (t.score == 0) { r.rescore_ok = r.wellformed_ok = true; }
        }
    }
    free_all();
#endif

    for (const auto& r : out.results) {
        if (r.resolved) ++out.resolved_count; else ++out.unresolved_count;
    }
    return out;
}

SWAlignResult align_sw(const SWRequest& req)
{
    std::vector<SWRequest> one{req};
    SWBatchResult b = align_sw_batch(one);
    return b.results.empty() ? SWAlignResult{} : b.results[0];
}

}  // namespace genoaligner
