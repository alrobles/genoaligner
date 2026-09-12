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
// The score kernel is the flat (fixed-block) one from Fase 7 when WFA_API_USE_FLAT
// is defined, otherwise the original. The flat variant was measured at 1.5-4.4x on
// MI210/PRO 6000 and is documented in docs/RESULTADO_FASE7_OPTIMIZACION.md; it is
// behind a flag rather than default because it is still a candidate, not merged.

#include "include/genoaligner/api.hpp"

#include "include/genoaligner/backend/wfa_kernel.hip"
#include "include/genoaligner/backend/wfa_score_flat.hip"

#include <cstdio>
#include <cstring>

#ifdef GENOALIGNER_HIP_SHIM
// The shim declares the launch-config globals extern and someone must define them.
// On the GPU paths the runtime provides them; under the shim this translation unit
// is the "launch owner", so it defines them -- required for the link to succeed even
// though the shim branch below never launches. Same arrangement as
// tests/parity/wfa_parity.cpp.
uint3 threadIdx{0, 0, 0};
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

static LaunchCfg cfg_for(int smax, bool flat)
{
    LaunchCfg c{};
    const int need = 2 * smax + 1;
    int b = 1;
    while (b < need) b <<= 1;
    // DO NOT clamp silently. The kernel maps diagonal k = threadIdx.x - s, so a
    // block smaller than 2*smax+1 skips diagonals WITHOUT RAISING AN ERROR -- the
    // kernel header says so explicitly. Clamping here (the first version did) would
    // therefore turn a legitimate request for a large smax into quietly wrong
    // alignments. smax is validated in align_batch before this is reached; the
    // capping branch survives only as a defensive assertion that must never fire.
    if (b > 1024) b = 1024;
    c.block = b;
    if (flat) c.block = WFA_FLAT_BLOCK;
    c.shmem = (size_t)2 * (2 * smax + 3) * sizeof(int);
    return c;
}

// The largest smax the score kernel can cover with a 1024-thread block. Above this
// the thread-to-diagonal mapping cannot cover [0, 2s] and the kernel silently skips
// diagonals, so the API REFUSES rather than degrading. 511 = (1024 - 1) / 2.
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
    hipMemcpy(d_pairs, views.data(), (size_t)N * sizeof(PairView), hipMemcpyHostToDevice);

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
        hipMemset(d_ws, 0, (size_t)wf_alloc_max * (size_t)N * sizeof(int));

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
        hipDeviceSynchronize();

        std::vector<int> scores((size_t)N, -1), meta((size_t)N * 4, 0);
        std::vector<int> cg((size_t)N * (size_t)cigar_cap, 0);
        hipMemcpy(scores.data(), d_scores, (size_t)N * sizeof(int), hipMemcpyDeviceToHost);
        hipMemcpy(meta.data(), d_meta, (size_t)N * 4 * sizeof(int), hipMemcpyDeviceToHost);
        hipMemcpy(cg.data(), d_cg, (size_t)N * (size_t)cigar_cap * sizeof(int), hipMemcpyDeviceToHost);

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
        // Score-only path.
#ifdef WFA_API_USE_FLAT
        LaunchCfg c = cfg_for(smax, true);
        hipLaunchKernelGGL(wfa_score_kernel_flat, dim3((unsigned)N), dim3(c.block), c.shmem,
                           (hipStream_t)0, d_pairs, d_scores, smax);
#else
        LaunchCfg c = cfg_for(smax, false);
        hipLaunchKernelGGL(wfa_score_kernel, dim3((unsigned)N), dim3(c.block), c.shmem,
                           (hipStream_t)0, d_pairs, d_scores, smax);
#endif
        hipError_t le = hipGetLastError();
        if (le != hipSuccess) { return fail("wfa_score_kernel launch"); }
        hipDeviceSynchronize();
        std::vector<int> scores((size_t)N, -1);
        hipMemcpy(scores.data(), d_scores, (size_t)N * sizeof(int), hipMemcpyDeviceToHost);
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

}  // namespace genoaligner
