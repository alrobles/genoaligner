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
//     kernel (blockDim=1 makes it degenerate; documented in bench/h7_flat_parity.cpp),
//     so the API returns everything unresolved by design. This path verifies the
//     API's SHAPE only: packing, ordering, counters, and that it degrades honestly
//     instead of inventing results.
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

static void check(bool ok, const char* what)
{
    if (!ok) { printf("  FAIL: %s\n", what); ++g_fail; }
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

int main()
{
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
    printf("  NOTE: CPU-shim build. This path verifies API SHAPE only -- the shim\n");
    printf("        cannot score through the kernel. Correctness is the GPU path.\n\n");
#else
    const bool gpu_build = true;
#endif
    const bool gpu = gpu_build && device_available();
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
        check(b.results.size() == 5, "batch returns one result per request");
        check(b.resolved_count + b.unresolved_count == 5, "counters partition the batch");
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
            r.smax    = 128;
            reqs.push_back(r);
        }

        BatchResult b = align_batch(reqs);
        int mismatch = 0, resolved = 0, cigar_bad = 0;
        for (int c = 0; c < NP; ++c) {
            const AlignResult& r = b.results[(size_t)c];
            const int want = edit_distance_cpu(pats[(size_t)c].data(), (int)pats[(size_t)c].size(),
                                               texts[(size_t)c].data(), (int)texts[(size_t)c].size());
            if (!r.resolved) { if (want <= 128) ++mismatch; continue; }
            ++resolved;
            if (r.score != want) { ++mismatch; if (mismatch <= 5) printf("  score mismatch: api=%d cpu=%d\n", r.score, want); }
            if (!r.rescore_ok || !r.wellformed_ok) { ++cigar_bad; if (cigar_bad <= 5) printf("  cigar invalid on pair %d (%s)\n", c, r.cigar.c_str()); }
        }
        printf("  pairs=%d resolved=%d score mismatches=%d invalid cigars=%d\n",
               NP, resolved, mismatch, cigar_bad);
        check(mismatch == 0, "API scores match the CPU DP");
        check(cigar_bad == 0, "API CIGARs pass both validators");
        check(b.resolved_count == resolved, "resolved_count matches the per-result flags");
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
        check(b.ok(), "batch reported ok");
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
        check(b.ok() && b.error == nullptr, "success path leaves status ok and error null");
        printf("  ok=%d error=%s\n", (int)b.ok(), b.error ? b.error : "(null)");
    }

    // ---- input validation: bad input must be REFUSED, not degraded ------
    // smax > 511 makes the kernel skip diagonals silently, so the API must not
    // accept it: an out-of-range bound would produce wrong alignments, not an error.
    printf("\n-- input validation --\n");
    {
        AlignRequest r;
        r.pattern = "ACGT"; r.pattern_len = 4;
        r.text    = "ACGT"; r.text_len    = 4;

        r.smax = 512;
        BatchResult b1 = align_batch({r});
        check(b1.status == BatchResult::Status::invalid_argument, "smax=512 refused");
        printf("  smax=512   : ok=%d error=%s\n", (int)b1.ok(), b1.error ? b1.error : "(null)");

        r.smax = 511;
        BatchResult b2 = align_batch({r});
        check(b2.ok(), "smax=511 accepted (the documented maximum)");
        printf("  smax=511   : ok=%d\n", (int)b2.ok());

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
        printf("RESULT: PASS (shape only) -- shim path; correctness requires the GPU build.\n");
        return 0;
    }
    printf("RESULT: FAIL (%d)\n", g_fail);
    return 1;
}
