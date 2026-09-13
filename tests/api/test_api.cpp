// genoaligner — Fase 8: API test.
//
// WHAT THIS VERIFIES, AND ON WHICH PATH
// -------------------------------------
// The API has two build paths and they can prove different things:
//
//   GPU path (hipcc/nvcc): runs the REAL kernels. This is where the API's scores
//     and CIGARs are checked against the independent CPU DP, and where the example
//     in api.hpp is executed verbatim -- so the documented usage cannot drift from
//     the working usage.
//
//   CPU-shim path (g++ + GENOALIGNER_HIP_SHIM): the shim cannot score through the
//     WFA kernel (blockDim=1 makes it degenerate; documented in
//     bench/h7_flat_parity.cpp), so WFA requests return unresolved by design and
//     this path verifies the WFA API's SHAPE only. The SW API is different: the
//     SW kernels DO run under the shim (one thread per pair / warp emulation), so
//     SW correctness is checked for real on every build.
//
// Reporting which path ran is mandatory, not decorative: a green run of the shim
// path says nothing about correctness, and must not be read as if it did.
//
// Build (GPU):
//   hipcc -O2 -std=c++17 -I. -o /tmp/test_api tests/api/test_api.cpp src/api/api.cpp
// Build (shim):
//   g++ -O2 -std=c++17 -DGENOALIGNER_HIP_SHIM -Itests/parity/hip_cpu_shim -I. \
//       -o /tmp/test_api tests/api/test_api.cpp src/api/api.cpp

#include "include/genoaligner/api.hpp"
#include "src/reference/edit_distance_cpu.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace genoaligner;

static int g_fail = 0;
static bool g_dont_care = false;   // set when no device is visible: nothing that
                                   // depends on execution can be asserted (WFA)
static bool g_sw_exec = false;     // SW results are real under the shim too, so
                                   // SW correctness is checked unless this is a
                                   // GPU build with no device visible
static const char* g_sw_emit = nullptr;   // --emit-sw <tsv>: API results for the
                                        // SeqAn3 oracle on the cluster

static void check(bool ok, const char* what)
{
    if (!ok) { printf("  FAIL: %s\n", what); ++g_fail; }
}

// A check that needs real execution. On a host with no device there is nothing to
// verify, so it is neither passed nor failed -- it is not evaluated. Using plain
// check() for these reported a MISSING GPU as four separate library bugs, which is
// the third time this file has confused "cannot run here" with "is broken".
static void check_exec(bool ok, const char* what)
{
    if (g_dont_care) return;
    check(ok, what);
}

// Same rule for SW, with one difference: the SW kernels DO run under the CPU
// shim, so on a shim build g_sw_exec is true and these checks assert for real.
static void check_sw(bool ok, const char* what)
{
    if (!g_sw_exec) return;
    check(ok, what);
}

// ---------------------------------------------------------------------------
// SW: an independent score reference for this file only (a fourth writing of
// the same recurrence -- test_sw_trace.cpp and the SeqAn3 oracle are the other
// checks; duplicating keeps the comparison honest).
// ---------------------------------------------------------------------------
static int sw_score_cpu(const char* text, int m, const char* pattern, int n,
                        const SWScoring& s)
{
    if (m <= 0 || n <= 0) return 0;
    std::vector<int> h((size_t)n + 1, 0), f((size_t)n + 1, 0);
    int best = 0;
    for (int i = 1; i <= m; ++i) {
        int hleft = 0, e = 0, diag = 0;
        for (int j = 1; j <= n; ++j) {
            { const int t = hleft - s.gap_open; const int u = e - s.gap_extend;
              e = (u > t) ? u : t; }
            { const int t = h[(size_t)j] - s.gap_open; const int u = f[(size_t)j] - s.gap_extend;
              f[(size_t)j] = (u > t) ? u : t; }
            int v = diag + (text[i - 1] == pattern[j - 1] ? s.match : s.mismatch);
            if (e > v) v = e;
            if (f[(size_t)j] > v) v = f[(size_t)j];
            if (v < 0) v = 0;
            diag = h[(size_t)j];
            h[(size_t)j] = v;
            hleft = v;
            if (v > best) best = v;
        }
    }
    return best;
}

// The SW example from api.hpp, executed verbatim for the same reason as the
// WFA one above.
static void run_sw_documented_example()
{
    printf("-- documented SW example (verbatim from api.hpp) --\n");
    genoaligner::SWRequest req;
    req.text        = "TTTACGTGTT"; req.text_len    = 10;
    req.pattern     = "ACGTGT";     req.pattern_len = 6;
    req.scoring     = {2, -3, 5, 2};
    genoaligner::SWAlignResult r = genoaligner::align_sw(req);
    printf("  score=%d cigar=%s span t[%d..%d] p[%d..%d]\n",
           r.score, r.cigar.c_str(), r.start_i, r.end_i, r.start_j, r.end_j);
    check_sw(r.resolved && r.score == 12, "documented SW example: score 12");
    check_sw(r.cigar == "MMMMMM", "documented SW example: cigar MMMMMM");
    check_sw(r.start_i == 3 && r.end_i == 8 && r.start_j == 0 && r.end_j == 5,
             "documented SW example: span t[3..8] x p[0..5]");
    printf("\n");
}

// The example from api.hpp, executed verbatim. If this drifts from the header, the
// header is lying to its reader.
static void run_documented_example()
{
    printf("-- documented example (verbatim from api.hpp) --\n");
    AlignRequest req;
    req.pattern = "ACGTACGT"; req.pattern_len = 8;
    req.text    = "ACGTTCGT"; req.text_len    = 8;
    req.smax    = 32;
    AlignResult r = align(req);
    if (r.resolved) {
        printf("  score=%d cigar=%s\n", r.score, r.cigar.c_str());
    } else {
        printf("  unresolved (expected on the shim path)\n");
    }
    printf("\n");
}

int main(int argc, char** argv)
{
    for (int a = 1; a < argc; ++a)
        if (std::strcmp(argv[a], "--emit-sw") == 0 && a + 1 < argc)
            g_sw_emit = argv[++a];

    printf("genoaligner API test\n");
    printf("  backend : %s\n", backend_name());
    printf("  device  : %s\n", device_name());
    printf("  available : %s\n\n", device_available() ? "yes" : "no");

    // "Is this a GPU build" is a COMPILE-TIME question; "can I score here" is a
    // RUNTIME one. The first version conflated them: a hipcc-built binary run on a
    // node with no device (the login node, or a CI runner) set gpu = true and then
    // failed 10 correctness checks that could not possibly have run -- reporting a
    // broken library when the truth was "no accelerator present". Both conditions
    // must hold for the correctness checks to be meaningful.
#ifdef GENOALIGNER_HIP_SHIM
    const bool gpu_build = false;
    printf("  NOTE: CPU-shim build. WFA checks verify API SHAPE only (the shim\n");
    printf("        cannot score through the WFA kernel). SW checks run the real\n");
    printf("        kernels on host memory and assert correctness.\n\n");
#else
    const bool gpu_build = true;
#endif
    const bool gpu = gpu_build && device_available();
    g_dont_care = !gpu;
    // SW is real on every build except a GPU binary with no device: the shim
    // executes the actual SW kernels on host memory, so SW correctness is
    // assertable where WFA's is not.
    g_sw_exec = gpu || !gpu_build;
    if (gpu_build && !gpu) {
        printf("  NOTE: GPU build, but no device is visible here. Correctness checks\n");
        printf("        are SKIPPED (they would fail for the wrong reason). Run on a\n");
        printf("        node with a GPU to verify scores and CIGARs.\n\n");
    }

    run_documented_example();

    // ---- shape: ordering and counters, on every path ---------------------
    printf("-- batch shape --\n");
    {
        std::vector<AlignRequest> reqs(5);
        for (int i = 0; i < 5; ++i) {
            reqs[(size_t)i].pattern = "AAAA"; reqs[(size_t)i].pattern_len = 4;
            reqs[(size_t)i].text    = "AAAA"; reqs[(size_t)i].text_len    = 4;
            reqs[(size_t)i].smax    = 32;
        }
        BatchResult b = align_batch(reqs);
        check_exec(b.results.size() == 5, "batch returns one result per request");
        check_exec(b.resolved_count + b.unresolved_count == 5, "counters partition the batch");
        printf("  results=%zu resolved=%d unresolved=%d\n\n",
               b.results.size(), b.resolved_count, b.unresolved_count);
    }

    // ---- edge cases: must not crash, must be consistent ------------------
    printf("-- edge cases --\n");
    {
        AlignRequest e;
        e.pattern = ""; e.pattern_len = 0;
        e.text    = "ACGT"; e.text_len = 4;
        e.smax    = 8;
        AlignResult r = align(e);
        if (gpu) check(r.resolved && r.score == 4, "empty pattern vs 4 chars -> score 4");
        printf("  empty pattern : resolved=%d score=%d\n", (int)r.resolved, r.score);

        AlignRequest e2;
        e2.pattern = "ACGT"; e2.pattern_len = 4;
        e2.text    = "ACGT"; e2.text_len    = 4;
        e2.smax    = 8;
        AlignResult r2 = align(e2);
        if (gpu) {
            check(r2.resolved && r2.score == 0, "identical -> score 0");
            check(r2.cigar == "MMMM", "identical -> CIGAR MMMM");
            check(r2.rescore_ok && r2.wellformed_ok, "identical passes both validators");
        }
        printf("  identical     : resolved=%d score=%d cigar=%s rescore_ok=%d wf_ok=%d\n",
               (int)r2.resolved, r2.score, r2.cigar.c_str(),
               (int)r2.rescore_ok, (int)r2.wellformed_ok);
    }

    // ---- correctness against the independent CPU DP (GPU path only) -------
    if (gpu) {
        printf("\n-- scores vs CPU DP (independent reference) --\n");
        std::mt19937 rng(20260912u);
        std::uniform_int_distribution<int> base(0, 3), pos(0, 511), ch(0, 3);
        const char alpha[] = "ACGT";

        const int NP = 200;
        std::vector<AlignRequest> reqs;
        std::vector<std::string> pats, texts;
        reqs.reserve(NP); pats.reserve(NP); texts.reserve(NP);
        for (int c = 0; c < NP; ++c) {
            const int len = 8 + (c % 200);
            const int ident = (c % 3 == 0) ? 90 : (c % 3 == 1) ? 98 : 100;
            std::string t;
            for (int i = 0; i < len; ++i) t.push_back(alpha[base(rng)]);
            std::string p = t;
            const int nmut = (int)((size_t)len * (100 - ident) / 100.0);
            for (int i = 0; i < nmut && len > 0; ++i) p[pos(rng) % len] = alpha[ch(rng)];
            pats.push_back(p); texts.push_back(t);
        }
        for (int c = 0; c < NP; ++c) {
            AlignRequest r;
            r.pattern = pats[(size_t)c].data(); r.pattern_len = (int)pats[(size_t)c].size();
            r.text    = texts[(size_t)c].data(); r.text_len   = (int)texts[(size_t)c].size();
            // smax=64, not 128: the trace kernel's per-block shared workspace is
            // (smax+1)(2*smax+3) ints -- 129*259*4 = 134 KB at smax=128, past the
            // 64 KiB device limit, so with_cigar at smax=128 is refused BY
            // DESIGN (the API error says so). These pairs mutate only by
            // substitution (d <= ~21 for len <= 207), so 64 covers them.
            r.smax    = 64;
            reqs.push_back(r);
        }

        BatchResult b = align_batch(reqs);
        int mismatch = 0, resolved = 0, cigar_bad = 0;
        for (int c = 0; c < NP; ++c) {
            const AlignResult& r = b.results[(size_t)c];
            const int want = edit_distance_cpu(pats[(size_t)c].data(), (int)pats[(size_t)c].size(),
                                               texts[(size_t)c].data(), (int)texts[(size_t)c].size());
            if (!r.resolved) { if (want <= 64) ++mismatch; continue; }
            ++resolved;
            if (r.score != want) { ++mismatch; if (mismatch <= 5) printf("  score mismatch: api=%d cpu=%d\n", r.score, want); }
            if (!r.rescore_ok || !r.wellformed_ok) { ++cigar_bad; if (cigar_bad <= 5) printf("  cigar invalid on pair %d (%s)\n", c, r.cigar.c_str()); }
        }
        printf("  pairs=%d resolved=%d score mismatches=%d invalid cigars=%d\n",
               NP, resolved, mismatch, cigar_bad);
        check_exec(mismatch == 0, "API scores match the CPU DP");
        check_exec(cigar_bad == 0, "API CIGARs pass both validators");
        check_exec(b.resolved_count == resolved, "resolved_count matches the per-result flags");

        // Score-only re-run of the same pairs: this is the path that launches
        // the FLAT kernel (the merged default). Scores must match the CPU DP
        // exactly -- without this, the GPU run only ever checked the score
        // path's SHAPE, and the flat merge would have shipped verified only on
        // the shim.
        for (auto& r : reqs) r.with_cigar = false;
        BatchResult bs = align_batch(reqs);
        check_exec(bs.ok(), "score-only batch reported ok");
        int smism = 0;
        for (int c = 0; c < NP; ++c) {
            const AlignResult& r = bs.results[(size_t)c];
            if (!r.resolved) continue;
            const int want = edit_distance_cpu(pats[(size_t)c].data(), (int)pats[(size_t)c].size(),
                                               texts[(size_t)c].data(), (int)texts[(size_t)c].size());
            if (r.score != want) { ++smism; if (smism <= 5) printf("  score-only mismatch: api=%d cpu=%d\n", r.score, want); }
        }
        printf("  score-only path (flat kernel): mismatches=%d\n", smism);
        check_exec(smism == 0, "score-only (flat kernel) scores match the CPU DP");
    }

    // ---- heterogeneous batch lengths: the case that broke cigar_cap ------
    // The first version sized the CIGAR buffer from reqs[0] and applied it to the
    // whole batch, so every pair longer than the first got a truncated (empty) CIGAR
    // while its SCORE was correct. Scores right, CIGARs silently wrong is the most
    // dangerous shape of bug this project has, so the case is pinned here.
    printf("\n-- heterogeneous batch lengths (cigar_cap regression) --\n");
    {
        std::vector<AlignRequest> reqs;
        std::vector<std::string> pats, texts;
        const int lens[] = {8, 40, 120, 300, 55};
        for (int L : lens) {
            std::string t, p;
            for (int i = 0; i < L; ++i) { t.push_back("ACGT"[i % 4]); p.push_back("ACGT"[i % 4]); }
            p[L / 2] = (p[L / 2] == 'A') ? 'C' : 'A';   // exactly one substitution
            pats.push_back(p); texts.push_back(t);
        }
        for (size_t i = 0; i < pats.size(); ++i) {
            AlignRequest r;
            r.pattern = pats[i].data(); r.pattern_len = (int)pats[i].size();
            r.text    = texts[i].data(); r.text_len   = (int)texts[i].size();
            r.smax    = 64;
            reqs.push_back(r);
        }
        BatchResult b = align_batch(reqs);
        check_exec(b.ok(), "batch reported ok");
        int bad = 0;
        for (size_t i = 0; i < reqs.size(); ++i) {
            const AlignResult& r = b.results[i];
            if (!r.resolved) { ++bad; continue; }
            if (r.score != 1) { ++bad; printf("  pair %zu: score=%d want 1\n", i, r.score); }
            if (r.cigar.size() != (size_t)reqs[i].pattern_len) {
                ++bad;
                printf("  pair %zu: cigar len=%zu want %d\n", i, r.cigar.size(), reqs[i].pattern_len);
            }
            if (!r.rescore_ok || !r.wellformed_ok) { ++bad; printf("  pair %zu: validators failed\n", i); }
        }
        printf("  pairs=%zu bad=%d\n", reqs.size(), bad);
        if (gpu) check(bad == 0, "heterogeneous batch: scores and CIGAR lengths all correct");
        else     printf("  (shape only on the shim path: CIGARs are not produced here)\n");
    }

    // ---- status reporting: failure must not look like 'unresolved' -------
    printf("\n-- status field --\n");
    {
        std::vector<AlignRequest> reqs(2);
        for (auto& r : reqs) {
            r.pattern = "AC"; r.pattern_len = 2;
            r.text    = "AC"; r.text_len    = 2;
            r.smax    = 8;
        }
        BatchResult b = align_batch(reqs);
        check_exec(b.ok() && b.error == nullptr, "success path leaves status ok and error null");
        printf("  ok=%d error=%s\n", (int)b.ok(), b.error ? b.error : "(null)");
    }

    // ---- input validation: bad input must be REFUSED, not degraded ------
    // smax > 511 makes the kernel skip diagonals silently, so the API must not
    // accept it: an out-of-range bound would produce wrong alignments, not an error.
    //
    // NOTE these only assert on the accept path. The REJECT paths are checked on
    // every build (they return before touching a device), but "smax=511 is accepted"
    // requires a working device: without one the call fails at allocation for an
    // unrelated reason, and asserting acceptance there would report a device's
    // absence as a validation bug. That is exactly the confusion this file was
    // already fixed for once.
    printf("\n-- input validation --\n");
    {
        AlignRequest r;
        r.pattern = "ACGT"; r.pattern_len = 4;
        r.text    = "ACGT"; r.text_len    = 4;

        r.smax = 512;
        BatchResult b1 = align_batch({r});
        check(b1.status == BatchResult::Status::invalid_argument, "smax=512 refused");
        printf("  smax=512   : ok=%d error=%s\n", (int)b1.ok(), b1.error ? b1.error : "(null)");

        r.smax = -1;
        BatchResult b3 = align_batch({r});
        check(b3.status == BatchResult::Status::invalid_argument, "negative smax refused");
        printf("  smax=-1    : ok=%d error=%s\n", (int)b3.ok(), b3.error ? b3.error : "(null)");

        AlignRequest nullp;
        nullp.pattern = nullptr; nullp.pattern_len = 5;
        nullp.text = "ACGT"; nullp.text_len = 4; nullp.smax = 8;
        BatchResult b4 = align_batch({nullp});
        check(b4.status == BatchResult::Status::invalid_argument, "null pointer refused");
        printf("  null ptr   : ok=%d error=%s\n", (int)b4.ok(), b4.error ? b4.error : "(null)");

        // The boundary itself needs a device to be meaningful. Score-only:
        // with_cigar at smax=511 asks the trace kernel for ~2 MB of per-block
        // shared memory, which no current device accepts -- the API refuses it
        // BY DESIGN. What this check pins is that 511 is inside the kernel's
        // accepted bound, which the score kernel answers.
        r.smax = 511;
        r.with_cigar = false;
        BatchResult b2 = align_batch({r});
        if (gpu) {
            check_exec(b2.ok(), "smax=511 accepted (the documented maximum)");
            printf("  smax=511   : ok=%d\n", (int)b2.ok());
        } else {
            printf("  smax=511   : (not asserted -- needs a device)\n");
        }
    }

    // =======================================================================
    // SMITH-WATERMAN. Unlike WFA, the shim path computes REAL results, so the
    // correctness checks below assert on every build except a GPU binary with
    // no device visible.
    // =======================================================================
    run_sw_documented_example();

    // ---- known answers ---------------------------------------------------
    printf("-- SW known answers --\n");
    {
        const SWScoring sc{2, -3, 5, 2};
        {
            SWRequest r;
            r.text = "ACGTACGT"; r.text_len = 8;
            r.pattern = "ACGTACGT"; r.pattern_len = 8;
            r.scoring = sc;
            SWAlignResult a = align_sw(r);
            check_sw(a.resolved && a.score == 16, "identical 8mers: score 16");
            check_sw(a.cigar == "MMMMMMMM", "identical 8mers: 8M");
            check_sw(a.start_i == 0 && a.end_i == 7 && a.start_j == 0 && a.end_j == 7,
                     "identical 8mers: full span");
            check_sw(a.rescore_ok && a.wellformed_ok, "identical 8mers: validators");
            printf("  identical     : score=%d cigar=%s\n", a.score, a.cigar.c_str());
        }
        {
            // A gap run must appear as ONE affine run, not scattered opens:
            // text = 8-mer + 6-insert + 8-mer, pattern = the two 8-mers. The
            // flanks are long enough that paying one length-6 gap (5+5*2=15)
            // beats taking either flank alone (16), so the optimum is
            // 8M 6I 8M, score 2*16-15 = 17 -- verified against the independent
            // reference before pinning the string here.
            SWRequest r;
            const std::string t = "AACCGGTT" + std::string(6, 'C') + "TTGGCCAA";
            const std::string p = "AACCGGTTTTGGCCAA";
            r.text = t.data(); r.text_len = (int)t.size();
            r.pattern = p.data(); r.pattern_len = (int)p.size();
            r.scoring = sc;
            SWAlignResult a = align_sw(r);
            check_sw(a.resolved && a.score == 17, "6-base insertion: score 17");
            check_sw(a.cigar == "MMMMMMMMIIIIIIMMMMMMMM",
                     "6-base insertion: cigar 8M6I8M");
            check_sw(a.rescore_ok && a.wellformed_ok, "6-base insertion: validators");
            printf("  6I run        : score=%d cigar=%s\n", a.score, a.cigar.c_str());
        }
        {
            // Disjoint sequences: no positive alignment exists.
            SWRequest r;
            r.text = "AAAAAAAA"; r.text_len = 8;
            r.pattern = "CCCCCCCC"; r.pattern_len = 8;
            r.scoring = sc;
            SWAlignResult a = align_sw(r);
            check_sw(a.resolved && a.score == 0, "disjoint: score 0, resolved");
            check_sw(a.cigar.empty() && a.start_i == -1 && a.end_i == -1,
                     "disjoint: empty cigar, coords -1");
            check_sw(a.rescore_ok && a.wellformed_ok, "disjoint: vacuous validators");
            printf("  disjoint      : score=%d resolved=%d\n", a.score, (int)a.resolved);
        }
        {
            // Empty inputs are legal and resolve to the empty alignment.
            SWRequest r;
            r.text = "ACGT"; r.text_len = 4;
            r.pattern = nullptr; r.pattern_len = 0;
            r.scoring = sc;
            SWAlignResult a = align_sw(r);
            check_sw(a.resolved && a.score == 0 && a.cigar.empty(),
                     "empty pattern: resolved, score 0");
            printf("  empty pattern : resolved=%d score=%d\n", (int)a.resolved, a.score);
        }
        {
            // score-only: score + end coordinate, no CIGAR, no start coords.
            SWRequest r;
            r.text = "TTTACGTGTT"; r.text_len = 10;
            r.pattern = "ACGTGT"; r.pattern_len = 6;
            r.scoring = sc;
            r.with_cigar = false;
            SWAlignResult a = align_sw(r);
            check_sw(a.resolved && a.score == 12, "score-only: score 12");
            check_sw(a.cigar.empty() && a.start_i == -1 && a.start_j == -1,
                     "score-only: no cigar, no start coords");
            check_sw(a.end_i == 8 && a.end_j == 5, "score-only: end coords present");
            printf("  score-only    : score=%d end=(%d,%d)\n", a.score, a.end_i, a.end_j);
        }
    }

    // ---- random batch vs the independent CPU reference --------------------
    printf("\n-- SW batch vs independent CPU reference --\n");
    {
        const SWScoring sc{1, -1, 2, 1};
        std::mt19937 rng(20261129u);
        std::uniform_int_distribution<int> base(0, 3);
        const char alpha[] = "ACGT";

        const int NP = 120;
        std::vector<SWRequest> reqs;
        std::vector<std::string> texts, pats;
        for (int c = 0; c < NP; ++c) {
            const int m = (int)(rng() % 140), n = (int)(rng() % 140);
            std::string t, p;
            for (int i = 0; i < m; ++i) t.push_back(alpha[base(rng)]);
            for (int i = 0; i < n; ++i) p.push_back(alpha[base(rng)]);
            texts.push_back(t); pats.push_back(p);
        }
        for (int c = 0; c < NP; ++c) {
            SWRequest r;
            r.text = texts[(size_t)c].data(); r.text_len = (int)texts[(size_t)c].size();
            r.pattern = pats[(size_t)c].data(); r.pattern_len = (int)pats[(size_t)c].size();
            r.scoring = sc;
            reqs.push_back(r);
        }
        SWBatchResult b = align_sw_batch(reqs);
        check_sw(b.ok(), "SW batch reports ok");
        int mismatch = 0, bad_cigar = 0;
        for (int c = 0; c < NP; ++c) {
            const SWAlignResult& r = b.results[(size_t)c];
            const int want = sw_score_cpu(texts[(size_t)c].data(), (int)texts[(size_t)c].size(),
                                          pats[(size_t)c].data(), (int)pats[(size_t)c].size(), sc);
            if (!r.resolved) { ++mismatch; if (mismatch <= 5) printf("  pair %d unresolved (want %d)\n", c, want); continue; }
            if (r.score != want) { ++mismatch; if (mismatch <= 5) printf("  pair %d: api=%d cpu=%d\n", c, r.score, want); }
            if (!r.rescore_ok || !r.wellformed_ok) ++bad_cigar;
            if (r.score == 0 && (!r.cigar.empty() || r.start_i != -1)) ++bad_cigar;
            if (r.score > 0 && (r.start_i < 0 || r.start_i > r.end_i ||
                                r.end_i >= (int)texts[(size_t)c].size() ||
                                r.end_j >= (int)pats[(size_t)c].size())) ++bad_cigar;
        }
        printf("  pairs=%d mismatches=%d bad_cigars=%d resolved=%d/%d\n",
               NP, mismatch, bad_cigar, b.resolved_count, NP);
        check_sw(mismatch == 0, "SW API scores match the CPU reference");
        check_sw(bad_cigar == 0, "SW API CIGARs pass both validators");
        check_sw(b.resolved_count + b.unresolved_count == NP, "SW counters partition the batch");
    }

    // ---- heterogeneous batch: mixed sizes and a score-0 pair --------------
    printf("\n-- SW heterogeneous batch --\n");
    {
        const SWScoring sc{2, -3, 5, 2};
        std::vector<SWRequest> reqs;
        std::vector<std::string> texts, pats;
        const int lens[] = {8, 40, 120, 300, 55};
        for (int L : lens) {
            std::string t, p;
            for (int i = 0; i < L; ++i) { t.push_back("ACGT"[i % 4]); p.push_back("ACGT"[i % 4]); }
            texts.push_back(t); pats.push_back(p);
        }
        texts.push_back("AAAAAAAAAA"); pats.push_back("CCCCCCCCCC");   // score 0
        for (size_t i = 0; i < pats.size(); ++i) {
            SWRequest r;
            r.text = texts[i].data(); r.text_len = (int)texts[i].size();
            r.pattern = pats[i].data(); r.pattern_len = (int)pats[i].size();
            r.scoring = sc;
            reqs.push_back(r);
        }
        SWBatchResult b = align_sw_batch(reqs);
        int bad = 0;
        for (size_t i = 0; i < reqs.size(); ++i) {
            const SWAlignResult& r = b.results[i];
            const int want = sw_score_cpu(texts[i].data(), (int)texts[i].size(),
                                          pats[i].data(), (int)pats[i].size(), sc);
            if (!r.resolved || r.score != want) { ++bad; printf("  pair %zu: resolved=%d score=%d want %d\n", i, (int)r.resolved, r.score, want); }
            if (!r.rescore_ok || !r.wellformed_ok) { ++bad; printf("  pair %zu: validators failed\n", i); }
        }
        printf("  pairs=%zu bad=%d\n", reqs.size(), bad);
        check_sw(bad == 0, "SW heterogeneous batch: all scores and CIGARs correct");
    }

    // ---- the declared size limit ------------------------------------------
    printf("\n-- SW supported-size limit --\n");
    {
        // (m+1)*(n+1) > SW_MAX_TRACE_CELLS: 8193*8194 > 2^26.
        static const std::string big_t(8192, 'A'), big_p(8193, 'A');
        SWRequest r;
        r.text = big_t.data(); r.text_len = (int)big_t.size();
        r.pattern = big_p.data(); r.pattern_len = (int)big_p.size();
        SWAlignResult a = align_sw(r);
        check_sw(!a.resolved && a.too_large && a.score == -1,
                 "over-limit pair reports too_large, not a crash");
        printf("  8192x8193     : resolved=%d too_large=%d\n", (int)a.resolved, (int)a.too_large);
    }

    // ---- SW input validation: refuse, do not degrade ----------------------
    // All of these return before any device work, so they assert on EVERY
    // build -- like the WFA validation above.
    printf("\n-- SW input validation --\n");
    {
        SWRequest r;
        r.text = "ACGT"; r.text_len = 4;
        r.pattern = "ACGT"; r.pattern_len = 4;

        r.scoring = {0, -1, 2, 1};
        check(align_sw_batch({r}).status == SWBatchResult::Status::invalid_argument,
              "match=0 refused (degenerate)");
        r.scoring = {2, 2, 5, 2};
        check(align_sw_batch({r}).status == SWBatchResult::Status::invalid_argument,
              "mismatch >= match refused");
        r.scoring = {2, -3, 0, 2};
        check(align_sw_batch({r}).status == SWBatchResult::Status::invalid_argument,
              "gap_open=0 refused");
        r.scoring = {2, -3, 5, 0};
        check(align_sw_batch({r}).status == SWBatchResult::Status::invalid_argument,
              "gap_extend=0 refused");
        // The Fase B finding: ge > go is degenerate for the CIGAR but the
        // SCORE stays exact -- refused with_cigar, accepted score-only.
        r.scoring = {2, -3, 1, 2};
        check(align_sw_batch({r}).status == SWBatchResult::Status::invalid_argument,
              "ge > go with_cigar refused");
        r.with_cigar = false;
        SWBatchResult ge_only = align_sw_batch({r});
        check(ge_only.status == SWBatchResult::Status::ok,
              "ge > go score-only accepted (score stays exact)");
        check_sw(ge_only.ok() && ge_only.results[0].resolved && ge_only.results[0].score == 8,
                 "ge > go score-only: identical 4mers score 8");
        r.with_cigar = true;

        SWRequest nullp = r;
        nullp.pattern = nullptr; nullp.pattern_len = 5;
        check(align_sw_batch({nullp}).status == SWBatchResult::Status::invalid_argument,
              "null pointer refused");
        SWRequest neg = r;
        neg.text_len = -1;
        check(align_sw_batch({neg}).status == SWBatchResult::Status::invalid_argument,
              "negative length refused");

        // One launch, one scheme: mixed-scoring batches are refused.
        SWRequest other = r;
        other.scoring = {1, -1, 2, 1};
        check(align_sw_batch({r, other}).status == SWBatchResult::Status::invalid_argument,
              "mixed-scoring batch refused");
        printf("  (all refusals checked on every build)\n");
    }

    // ---- emit TSV for the SeqAn3 oracle (cluster leg) ----------------------
    if (g_sw_emit) {
        const SWScoring sc{1, -1, 2, 1};    // the oracle's scheme
        std::mt19937 rng(20261129u);
        std::uniform_int_distribution<int> base(0, 3);
        const char alpha[] = "ACGT";
        std::vector<SWRequest> reqs;
        std::vector<std::string> texts, pats;
        const int NP = 150;
        for (int c = 0; c < NP; ++c) {
            const int m = 4 + (int)(rng() % 200), n = 4 + (int)(rng() % 200);
            std::string t, p;
            for (int i = 0; i < m; ++i) t.push_back(alpha[base(rng)]);
            for (int i = 0; i < n; ++i) p.push_back(alpha[base(rng)]);
            texts.push_back(t); pats.push_back(p);
        }
        for (int c = 0; c < NP; ++c) {
            SWRequest r;
            r.text = texts[(size_t)c].data(); r.text_len = (int)texts[(size_t)c].size();
            r.pattern = pats[(size_t)c].data(); r.pattern_len = (int)pats[(size_t)c].size();
            r.scoring = sc;
            reqs.push_back(r);
        }
        SWBatchResult b = align_sw_batch(reqs);
        FILE* f = fopen(g_sw_emit, "w");
        if (f) {
            fprintf(f, "idx\ttext\tpattern\tscore\tstart_i\tstart_j\tend_i\tend_j\tcigar\n");
            for (int c = 0; c < NP; ++c) {
                const SWAlignResult& r = b.results[(size_t)c];
                fprintf(f, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%s\n", c,
                        texts[(size_t)c].c_str(), pats[(size_t)c].c_str(),
                        r.score, r.start_i, r.start_j, r.end_i, r.end_j,
                        r.cigar.c_str());
            }
            fclose(f);
            printf("--emit-sw: wrote %d API results to %s\n", NP, g_sw_emit);
        } else {
            printf("  --emit-sw: cannot open %s\n", g_sw_emit);
        }
    }

    printf("\n");
    if (g_fail == 0) {
        if (gpu) {
            printf("RESULT: PASS -- API verified on the GPU path against the CPU DP.\n");
            return 0;
        }
        if (gpu_build) {
            printf("RESULT: PASS (shape only) -- GPU build, no device visible here; "
                   "correctness NOT verified.\n");
            return 77;   // skipped: ctest treats 77 as skipped, not failed
        }
        printf("RESULT: PASS -- shim path; SW correctness verified here, "
               "WFA correctness requires the GPU build.\n");
        return 0;
    }
    printf("RESULT: FAIL (%d)\n", g_fail);
    return 1;
}
