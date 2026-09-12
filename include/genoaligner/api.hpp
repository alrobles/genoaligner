// genoaligner — public API.
//
// WHAT THIS IS
// ------------
// The consumption interface. Until this existed, the kernels could only be reached
// from the parity/bench harnesses, which is why integration into another project
// (phylogenyAI) was evaluated as blocked: there was nothing to call. This header is
// that thing.
//
// WHAT IT IS NOT
// --------------
//   - Not a multiple aligner. This aligns PAIRS. It does not replace MAFFT/MACSE.
//   - Not a search tool. It aligns two sequences you already have; it does not
//     recruit candidates from a database.
//   - Not a FASTQ/FASTA reader. Inputs are raw char buffers + lengths, so the API
//     has no file-format opinion and no IO dependency.
//
// BACKEND SELECTION
// -----------------
// The implementation is compiled once per backend (ROCm or CUDA) from the same
// source tree; the caller does not choose a backend at runtime. See
// docs/DEPLOYMENT_MANIFEST.md for the build recipes. `backend_name()` reports which
// one this binary is, so a caller can log it rather than guess.
//
// COST MODEL, because callers need it
// -----------------------------------
// The kernels are bounded by `smax` (max edit distance). Pairs whose true distance
// exceeds it are UNRESOLVED, not failed: the result carries score = -1 and no CIGAR.
// Size smax to your data or you will silently get a fraction of your pairs back.
// `align_batch` reports how many resolved, so the fraction is never a mystery.
//
// EXAMPLE
// -------
//     genoaligner::AlignRequest req;
//     req.pattern = "ACGTACGT"; req.pattern_len = 8;
//     req.text    = "ACGTTCGT"; req.text_len    = 8;
//     req.smax    = 32;
//     genoaligner::AlignResult r = genoaligner::align(req);
//     if (r.resolved) { printf("score=%d cigar=%s\n", r.score, r.cigar.c_str()); }
//
// The example is compiled and run by tests/api/test_api.cpp, so the header and the
// documented usage cannot drift apart.

#ifndef GENOALIGNER_API_HPP
#define GENOALIGNER_API_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace genoaligner {

// ---------------------------------------------------------------------------
// Result of aligning ONE pair.
// ---------------------------------------------------------------------------
struct AlignResult {
    int         score    = -1;   // edit distance; -1 = unresolved (distance > smax)
    bool        resolved = false; // score >= 0
    std::string cigar;           // over {M,X,I,D}; empty when unresolved

    // Validation flags, computed in the library rather than trusted from the
    // caller: re-score of the CIGAR against the score, and well-formedness
    // (consumes exactly pattern_len and text_len, M only on matches). These are
    // the same two checks the Fase 4 gate applies, so a result that comes out of
    // the API has already passed them.
    bool        rescore_ok     = false;
    bool        wellformed_ok  = false;
};

// ---------------------------------------------------------------------------
// One pair to align.
// ---------------------------------------------------------------------------
struct AlignRequest {
    const char* pattern     = nullptr;
    int         pattern_len = 0;
    const char* text        = nullptr;
    int         text_len    = 0;
    int         smax        = 64;    // max edit distance searched
    bool        with_cigar  = true;  // false = score only (cheaper, no traceback)
};

// ---------------------------------------------------------------------------
// THREAD SAFETY — the measured contract
// -------------------------------------
// Verified by tests/concurrency/test_concurrency.cpp: concurrent align_batch()
// calls on DISJOINT inputs return correct results (4 threads, 48 pairs, all checked
// against the CPU DP). That is what was measured, and the limits are:
//
//   SAFE      concurrent calls with disjoint inputs, from different threads.
//   UNSAFE    concurrent calls sharing a request buffer that another thread mutates.
//             The API stores POINTERS (const char*), it does not copy them, so the
//             caller must keep the input alive and unmodified until the call returns.
//             This is the most likely way to misuse this API.
//   UNSAFE    anything relying on launch ORDER. Kernels go to the default stream, so
//             two concurrent calls are ordered arbitrarily with respect to each other.
//             Correct results do not depend on order, but timing does -- do not read
//             throughput numbers from concurrent calls.
//
// These are backed by the test above and by reading the implementation, not by
// assumption. `device_name()` used to memoise through an unsynchronised
// check-then-write on a static buffer; it now uses a thread-safe function-local
// static, so calling it concurrently is fine.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Single-pair entry point. Convenient; for bulk work use align_batch, which
// amortises device setup across pairs.
// ---------------------------------------------------------------------------
AlignResult align(const AlignRequest& req);

// ---------------------------------------------------------------------------
// Batch entry point.
//
// Results come back in the SAME ORDER as the requests. `resolved_count` reports how
// many pairs had a true distance within smax; the rest are unresolved by design and
// carry score = -1. Reporting that count is deliberate: a caller that only looks at
// per-pair scores can miss that most of its input was abandoned, which is the
// failure mode this project spent a phase learning to make visible.
//
// ONE smax FOR THE WHOLE BATCH, AND IT IS THE MINIMUM
// ---------------------------------------------------
// The kernel takes a single smax, so a batch cannot have per-request bounds. The
// batch therefore uses the MINIMUM of the requests' smax values, and this is a
// documented contract, not an implementation detail:
//
//   - Minimum, never maximum. A per-request smax is a limit the CALLER set. Silently
//     granting more would resolve pairs the caller expected to be abandoned, which
//     is a behaviour change they cannot see. Taking the minimum can only under-serve,
//     which is visible (resolved_count drops) and never wrong.
//   - The cost is real: ONE request with a small smax lowers the bound for every pair
//     in the batch. If your batch mixes bounds, group by smax and make several calls
//     -- that is the supported way to get different bounds.
//   - If you want the widest bound, do not put a narrow request in the same batch.
//
// `smax` must be in [0, 511] (see kMaxSmax in the implementation); out-of-range
// values on ANY request fail the whole batch with Status::invalid_argument rather
// than being clamped, because clamping a bound silently skips diagonals in the
// kernel and would return wrong alignments.
// ---------------------------------------------------------------------------
struct BatchResult {
    std::vector<AlignResult> results;
    int resolved_count  = 0;
    int unresolved_count = 0;

    // Failure is NOT the same as "unresolved", and the first version of this API
    // conflated them: an allocation failure returned the same score = -1 that means
    // "true distance exceeded smax", so a caller could not tell a broken run from a
    // legitimate one. That is the failure mode this project spent a phase learning
    // to make visible, so the API reports it explicitly.
    enum class Status { ok, invalid_argument, device_error };
    Status status = Status::ok;
    const char* error = nullptr;   // static string, non-null iff status != ok

    bool ok() const { return status == Status::ok; }
};
BatchResult align_batch(const std::vector<AlignRequest>& reqs);

// ---------------------------------------------------------------------------
// Environment / provenance. Cheap calls, no device work.
// ---------------------------------------------------------------------------
const char* backend_name();      // "rocm" or "cuda", fixed at compile time
const char* device_name();       // device 0 as reported by the runtime, or "none"
bool        device_available();  // false when built/run without a visible GPU

}  // namespace genoaligner

#endif  // GENOALIGNER_API_HPP
