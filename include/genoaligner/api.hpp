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
//   - Not a search tool. It aligns sequences you already have; it does not
//     recruit candidates from a database.
//   - Not a FASTQ/FASTA reader. Inputs are raw char buffers + lengths (pairwise)
//     or std::strings (MSA), so the API has no file-format opinion and no IO
//     dependency.
//
// TWO PRODUCTS, ONE CORE
// ----------------------
//   - Pairwise: align / align_batch (bounded edit distance) and align_sw /
//     align_sw_batch (Smith-Waterman). For pairs you already hold.
//   - Multiple: msa_align (progressive profile-profile alignment; DNA, protein
//     and MACSE-class codon-aware modes). For a set of sequences.
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

// ===========================================================================
// SMITH-WATERMAN — local alignment (a different algorithm, a different meaning)
// ===========================================================================
//
// WHY SEPARATE ENTRY POINTS, NOT AN `algorithm` FIELD ON AlignRequest
// ------------------------------------------------------------------
// SWAlignResult::score is an alignment SCORE (higher is better, >= 0 always).
// AlignResult::score is an edit DISTANCE (lower is better). Overloading one
// field with both meanings is exactly the silent-semantic-change bug this
// project exists to avoid, so SW gets its own request/result types:
//
//   * SWRequest has NO smax. Smith-Waterman is not bounded by a distance
//     budget; there is nothing for the field to mean, so it is absent rather
//     than ignored.
//   * SWAlignResult carries the alignment's COORDINATES (start/end in both
//     sequences), which local alignment produces and WFA does not.
//   * A batch is homogeneous by type: you cannot accidentally mix distance
//     requests and score requests into one result vector.
//
// THE SCORING SCHEME
// ------------------
// SWRequest::scoring is {match, mismatch, gap_open, gap_extend}:
//   a match adds match; a mismatch adds mismatch; a gap run of length L
//   subtracts gap_open + (L-1)*gap_extend. The defaults below are an EXAMPLE,
//   not a recommendation -- pick the scheme for your data.
//
//   Validation REFUSES a request rather than computing under a degenerate
//   scheme (invalid_argument, whole batch fails):
//     match <= 0            SW is degenerate: the score is 0 for every input
//     mismatch >= match     a scheme where mismatching never loses to matching
//                           is pathological -- almost surely a caller error
//     gap_open <= 0 or
//     gap_extend <= 0       free or rewarded gaps are degenerate
//     gap_extend > gap_open  WITH with_cigar: under this regime the DP prefers
//                           re-opening adjacent 1-gaps over extending a run
//                           (two opens cost 2*gap_open < gap_open+gap_extend),
//                           and consecutive same-direction ops are ONE run in
//                           CIGAR terms -- the emitted string cannot reproduce
//                           its own score. Found empirically in Fase B
//                           (docs/RESULTADO_H9_SW2_TRACE.md). Score-only
//                           requests still accept it: the SCORE remains exact.
//     mixed scoring in one
//     batch                 one kernel launch takes ONE scheme, like the
//                           one-smax rule: group by scheme, call per scheme.
//
// COORDINATES AND THE ALIGNED SPAN
// --------------------------------
// A local alignment covers a SPAN, not the whole sequences. start_i/start_j
// are the 0-based indices of the first aligned char of text/pattern;
// end_i/end_j the last. The CIGAR covers exactly text[start_i..end_i] and
// pattern[start_j..end_j]. score == 0 means "no positive-scoring alignment
// exists" (disjoint sequences); all four coords are -1 and the CIGAR is empty.
// Score-only results (with_cigar = false, unmixed batch) report score and the
// end coordinate; the start coords are -1 because sw_score_kernel does not
// compute them.
//
// SUPPORTED-SIZE LIMITS, DECLARED
// -------------------------------
//   * with_cigar needs (text_len+1)*(pattern_len+1) bytes of direction table
//     per pair. Pairs whose matrix exceeds SW_MAX_TRACE_CELLS come back
//     resolved=false with too_large=true -- a reported limit, not a crash.
//   * with_cigar=false on an unmixed batch runs the warp kernel, whose
//     pattern_len is bounded by device shared memory (per-warp 2*(n+1) ints,
//     4 warps/block; ~2047 bases on a 64 KiB limit). A batch exceeding it is
//     refused with invalid_argument naming the limit.
//   * The whole batch's workspace is one device allocation; if it does not
//     fit in device memory the batch reports device_error. Split the batch.
//
// EXAMPLE
// -------
//     genoaligner::SWRequest req;
//     req.text        = "TTTACGTGTT"; req.text_len    = 10;
//     req.pattern     = "ACGTGT";     req.pattern_len = 6;
//     req.scoring     = {2, -3, 5, 2};
//     genoaligner::SWAlignResult r = genoaligner::align_sw(req);
//     // r.score == 12, cigar "MMMMMM", text[3..8] aligned to pattern[0..5]
//
// ---------------------------------------------------------------------------
// The scoring scheme. Penalties are positive and SUBTRACTED.
// ---------------------------------------------------------------------------
struct SWScoring {
    int match      = 2;
    int mismatch   = -3;
    int gap_open   = 5;
    int gap_extend = 2;
};

// The largest direction table a pair may need for traceback, in cells
// ((text_len+1)*(pattern_len+1)); ~8k x 8k pairs fit, ~67 MB per pair of
// device workspace. Larger pairs report too_large rather than overrun.
static constexpr int SW_MAX_TRACE_CELLS = 1 << 26;

// ---------------------------------------------------------------------------
// One pair to locally align.
// ---------------------------------------------------------------------------
struct SWRequest {
    const char* text        = nullptr;   // the row axis
    int         text_len    = 0;
    const char* pattern     = nullptr;   // the column axis
    int         pattern_len = 0;
    SWScoring   scoring;
    bool        with_cigar  = true;      // false = score + end coordinate only
};

// ---------------------------------------------------------------------------
// Result of locally aligning ONE pair.
// ---------------------------------------------------------------------------
struct SWAlignResult {
    int         score    = -1;   // alignment score, >= 0 when resolved;
                                 // -1 only when the pair was not computed
    bool        resolved = false;
    std::string cigar;           // over the aligned span; empty when score == 0

    int         start_i = -1;    // first aligned char of text    (-1 if score 0
    int         start_j = -1;    //   of pattern                    or unknown)
    int         end_i   = -1;    // last aligned char of text
    int         end_j   = -1;    //   of pattern

    // Same discipline as AlignResult: computed in the library, not trusted.
    // rescore_ok = the CIGAR re-scores to `score` under the affine scheme;
    // wellformed_ok = it consumes exactly the reported span, M on equal chars,
    // X on differing. score == 0 resolves with both true (vacuous).
    bool        rescore_ok    = false;
    bool        wellformed_ok = false;

    // The pair's matrix exceeded SW_MAX_TRACE_CELLS: a declared supported-size
    // limit, reported per pair. resolved is false and score is -1.
    bool        too_large = false;
};

// ---------------------------------------------------------------------------
// Batch entry point. Results come back in request order; resolved_count counts
// computed pairs (SW has no "unresolved by design" -- unresolved here means
// too_large or an execution failure, which is why the count exists).
// ---------------------------------------------------------------------------
struct SWBatchResult {
    std::vector<SWAlignResult> results;
    int resolved_count   = 0;
    int unresolved_count = 0;

    enum class Status { ok, invalid_argument, device_error };
    Status status = Status::ok;
    const char* error = nullptr;   // static string, non-null iff status != ok

    bool ok() const { return status == Status::ok; }
};
SWBatchResult align_sw_batch(const std::vector<SWRequest>& reqs);
SWAlignResult align_sw(const SWRequest& req);

// ===========================================================================
// MULTIPLE SEQUENCE ALIGNMENT — progressive profile-profile (genomsa engine)
// ===========================================================================
//
// WHAT THIS IS
// ------------
// Progressive MSA: fragment-corrected k-mer distances -> deterministic
// neighbor-joining guide tree -> post-order profile-vs-profile Gotoh DP
// (semiglobal, free end gaps) -> deterministic column-interleave merge.
// Position-specific gap penalties (ClustalW/TWILIGHT convention) and a
// gappy-column heuristic are ON by default. The pipeline is deterministic:
// NJ ties break on smaller index, DP ties prefer M > Ix > Iy, merge order
// follows the guide tree's left/right order. This entry runs the HOST
// engine: pure CPU, no device needed, same result on every platform.
//
// MODES (MsaRequest::mode)
// ------------------------
//   dna      alpha=4. IUPAC nucleotides; match/transition/transversion
//            scoring. Input must be unaligned IUPAC DNA (no '-').
//   protein  alpha=20. Amino acids, BLOSUM62 column scoring
//            (B/Z/J -> their two-letter sets, X/O uniform, U -> C, '*' and
//            unknown letters count as gap).
//   codon    alpha=65. MACSE-class codon-aware alignment: each sequence is
//            tokenized to codons, aligned over a 65x65 matrix (BLOSUM62 of
//            the translated amino acids + codon_nt_bonus per identical nt
//            - codon_stop_pen when exactly one side is a stop), then decoded
//            back to nucleotides. Indels are whole codons, so the output
//            keeps reading frame by construction. gc_def selects the NCBI
//            table: 1 = standard, 2 = vertebrate mitochondrial (CYTB/COI/ND).
//
//   codon_local_frame (default ON, codon only): a stop-avoiding DP
//            partitions each sequence into codon blocks and 1-2 nt
//            frameshift blocks, so rows with internal frameshifts keep
//            downstream tokens in-frame. It implies codon_refine >= 1 (the
//            refine pass restores the real nts where placeholder tokens
//            sat), exactly like the driver's --local-frame default.
//   codon_refine (codon only): stage-2 refinement passes that reintroduce
//            frameshift events as 1-2 nt columns (MACSE '!'). With refine=0
//            AND local_frame=false the decoded rows are whole-codon
//            (width %3 == 0); any other codon configuration yields a NT MSA
//            that is NOT guaranteed %3 -- that is MACSE semantics, not a bug.
//
// QC, THE SAME DISCIPLINE AS PairwiseResult FLAGS
// -----------------------------------------------
// qc carries one entry per INPUT sequence, in input order; fields are -1 in
// non-codon modes (no frame concept exists) and measured values in codon
// mode: the chosen frame, the stop-codon count in that frame, and the
// partial/ambiguous codon count. The counts are computed in the library
// during tokenization -- a caller never has to guess which sequences were
// problematic going in.
//
// VALIDATION — refuses rather than guesses
// ----------------------------------------
// Whole request fails with Status::invalid_argument (no partial output):
//   no sequences, or any empty sequence      engine cannot index a tree on
//                                            nothing / an empty profile is
//                                            almost surely a caller bug
//   character outside the mode's alphabet    dna/codon: IUPAC DNA letters
//                                            (ACGTUN + ambiguity, no '-');
//                                            protein: A-Z + '*' and '?'
//   mode outside {dna,protein,codon}         a casted enum must not be
//                                            answered as if it were dna
//   gc_def not in {1,2}                      only those tables exist
//   gc_def != 1 outside codon mode           the field means nothing there;
//                                            an ignored field is a silent
//                                            contract break
//   codon_refine != 0 outside codon mode     same ignored-field rule
//   codon_refine < 0                         nonsensical round count
//
// ONE SEQUENCE is legal and returns the sequence itself (degenerate tree).
// aligned.size() == seqs.size(), all rows equal width, input order kept --
// verified before the result leaves the library, not trusted from the engine.
//
// TWO ENGINES, ONE RESULT
// -----------------------
//   msa_align(req)         the HOST engine described above. Portable: pure
//                          CPU, no device needed, same answer everywhere.
//   msa_align_device(req)  the DEVICE engine: NJ guide tree on the device
//                          (nj_tree_gpu), one profile-profile kernel launch
//                          per guide-tree level, host merge. The level batch
//                          is gated BIT-EXACT against the sequential engine
//                          (tests/parity/msa_driver_parity.cpp), so the rows
//                          are identical -- what changes is throughput and
//                          which processor did the work. MsaResult::device
//                          records which engine produced the rows.
//
// DEVICE SEMANTICS — requested device, or a refusal
// -------------------------------------------------
// msa_align_device answers Status::device_error (not a host result) when the
// device path fails -- no device visible, allocation failure, kernel error.
// Falling back to the host engine would silently serve a different
// PERFORMANCE contract: a device caller on 8k sequences asked for hours, not
// days. Call msa_align() yourself if host timing is acceptable.
// One exception, documented rather than hidden: in codon mode the stage-2
// refine may fall back to host codon_refine after a device failure, because
// its DP body is shared source between codon_refine and codon_refine_kernel
// -- identical answer either way, and the O(n*W) refine is not why a caller
// chose the device. The align core never falls back.
// device_error is a distinct Status, same convention as the pairwise API.
//
// THREAD SAFETY
// -------------
// msa_align is host code, safe for concurrent calls with disjoint requests.
// msa_align_device launches on the default stream: results do not depend on
// launch order, but concurrent calls serialise arbitrarily with respect to
// each other -- same caveat as the pairwise API.
//
// EXAMPLE
// -------
//     genoaligner::MsaRequest req;
//     req.mode = genoaligner::MsaMode::codon;
//     req.gc_def = 2;                    // vertebrate mitochondrial
//     req.seqs = {"ATGATAATCACC", "ATGATTATCACCTGA"};
//     genoaligner::MsaResult r = genoaligner::msa_align(req);
//     // r.aligned has 2 rows of equal width; every indel is a whole codon
//     // and r.qc[0].frame / r.qc[0].stops report the encode QC.
//
// The example is compiled and run by tests/api/test_msa_api.cpp, so the
// header and the documented usage cannot drift apart.
// ---------------------------------------------------------------------------
enum class MsaMode { dna, protein, codon };

// Per-input-sequence quality metrics. All fields are -1 outside codon mode.
struct MsaSeqQc {
    int frame   = -1;   // chosen forward frame, 0..2; trivially 0 under
                        // codon_local_frame (the DP places blocks, not one frame)
    int stops   = -1;   // stop codons in the tokenized stream
    int partial = -1;   // partial/ambiguous codons + frameshift blocks held
                        // out of the token stream
};

struct MsaRequest {
    std::vector<std::string> seqs;
    MsaMode mode = MsaMode::dna;
    int  gc_def            = 1;      // codon only: NCBI table 1 or 2
    int  codon_refine      = 0;      // codon only: stage-2 passes
    bool codon_local_frame = true;   // codon only: local-frame encode DP
};

struct MsaResult {
    std::vector<std::string> aligned;   // equal-length rows, input order
    std::vector<MsaSeqQc>    qc;        // one entry per input seq
    int width = 0;                      // column count (0 when empty)
    bool device = false;                // rows came from the device engine

    enum class Status { ok, invalid_argument, device_error, error };
    Status status = Status::ok;
    const char* error = nullptr;        // static string, non-null iff != ok

    bool ok() const { return status == Status::ok; }
};
MsaResult msa_align(const MsaRequest& req);
MsaResult msa_align_device(const MsaRequest& req);

// ---------------------------------------------------------------------------
// Environment / provenance. Cheap calls, no device work.
// ---------------------------------------------------------------------------
const char* backend_name();      // "rocm" or "cuda", fixed at compile time
const char* device_name();       // device 0 as reported by the runtime, or "none"
bool        device_available();  // false when built/run without a visible GPU

}  // namespace genoaligner

#endif  // GENOALIGNER_API_HPP
