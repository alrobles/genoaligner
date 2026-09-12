// genoaligner — B2: alignment on REAL biological sequences.
//
// WHY THIS EXISTS SEPARATELY FROM THE SYNTHETIC TESTS
// ---------------------------------------------------
// Every other correctness test in this project generates its input: random bases,
// or a random sequence with a controlled number of substitutions. That covers the
// algebra and nothing else. Real genomic sequence has properties a generator does not
// produce, and those are exactly where alignment implementations break:
//
//   - tandem repeats and homopolymer runs (the D-loop, microsatellites): the optimal
//     alignment is ambiguous, many CIGARs tie, and a traceback that breaks ties
//     badly can emit a CIGAR that does not reconstruct the sequence
//   - tRNA genes: short, GC-rich, highly structured, full of short internal matches
//   - low-complexity and single-base runs: where a wrong offset in the walk shows up
//   - N and lowercase soft-masking: real files contain them
//   - divergent-but-related homologs (human vs chimp mtDNA, ~88% identity): the
//     distances are large relative to the sequence length, so smax actually binds
//
// DATA PROVENANCE (do not replace with hand-written strings)
// ---------------------------------------------------------
//   tests/data/mtdna_human.fa   NC_012920.1  Homo sapiens mitochondrion, 16569 bp
//   tests/data/mtdna_chimp.fa   NC_001643.1  Pan troglodytes mitochondrion, 16554 bp
// Fetched from NCBI E-utilities (efetch, nuccore) on 2026-09-12. Human mtDNA contains
// one genuine N; chimp is pure ACGT. Both files are single-record, unwrapped.
//
// The reference is the independent CPU dynamic program, same as everywhere else.
// This test does not invent a new oracle; it changes the INPUT.

#include "include/genoaligner/api.hpp"
#include "include/genoaligner/io/fasta.hpp"
#include "src/reference/edit_distance_cpu.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace genoaligner;

static int  g_fail  = 0;
static bool g_ran   = false;   // did any pairwise assertion actually execute?

static void check(bool ok, const char* what)
{
    if (!ok) { printf("  FAIL: %s\n", what); ++g_fail; }
}

// Align every (pattern, text) pair and compare against the CPU DP.
// Returns the number of pairs compared.
static int compare_all(const std::vector<std::string>& pats,
                       const std::vector<std::string>& texts,
                       int smax, const char* label, int max_print = 4)
{
    if (pats.size() != texts.size()) { check(false, "test bug: unequal pair lists"); return 0; }

    std::vector<AlignRequest> reqs;
    reqs.reserve(pats.size());
    for (size_t i = 0; i < pats.size(); ++i) {
        AlignRequest r;
        r.pattern = pats[i].data(); r.pattern_len = (int)pats[i].size();
        r.text    = texts[i].data(); r.text_len    = (int)texts[i].size();
        r.smax    = smax;
        reqs.push_back(r);
    }
    BatchResult b = align_batch(reqs);
    if (!b.ok()) { printf("  batch failed: %s\n", b.error ? b.error : "?"); check(false, label); return 0; }

    int n_resolved = 0, n_bad = 0, n_validator = 0, n_empty = 0, printed = 0;
    for (size_t i = 0; i < reqs.size(); ++i) {
        const AlignResult& r = b.results[i];
        const int want = edit_distance_cpu(pats[i].data(), (int)pats[i].size(),
                                           texts[i].data(), (int)texts[i].size());
        if (!r.resolved) {
            // Unresolved is legitimate only if the true distance exceeds smax.
            if (want <= smax) {
                ++n_bad;
                if (printed++ < max_print)
                    printf("  UNRESOLVED but d=%d <= smax=%d  (pair %zu, lens %zu/%zu)\n",
                           want, smax, i, pats[i].size(), texts[i].size());
            }
            continue;
        }
        ++n_resolved;
        if (r.score != want) {
            ++n_bad;
            if (printed++ < max_print)
                printf("  score %d, CPU DP says %d  (pair %zu)\n", r.score, want, i);
        }
        // The CIGAR must reconstruct the pair AND re-score to the reported score.
        if (r.cigar.empty()) {
            ++n_empty;
            if (printed++ < max_print) printf("  EMPTY cigar on a resolved pair %zu\n", i);
        } else if (!r.rescore_ok || !r.wellformed_ok) {
            ++n_validator;
            if (printed++ < max_print)
                printf("  cigar invalid on pair %zu (rescore=%d wf=%d, len=%zu)\n",
                       i, (int)r.rescore_ok, (int)r.wellformed_ok, r.cigar.size());
        }
    }

    printf("  %-34s pairs=%zu resolved=%d bad=%d empty_cigar=%d invalid_cigar=%d\n",
           label, reqs.size(), n_resolved, n_bad, n_empty, n_validator);
    g_ran = true;
    check(n_bad == 0, label);
    check(n_empty == 0, "no empty CIGARs on resolved pairs");
    check(n_validator == 0, "all CIGARs pass re-score and well-formedness");
    return (int)reqs.size();
}

static std::string slice(const std::string& s, size_t from, size_t len)
{
    if (from >= s.size()) return std::string();
    return s.substr(from, std::min(len, s.size() - from));
}

int main()
{
    printf("genoaligner — real-sequence alignment test (B2)\n");
    printf("  backend : %s\n", backend_name());
    printf("  device  : %s\n\n", device_name());

    // ---- load the real data ----------------------------------------------
    std::vector<io::FastaRecord> hrec, crec;
    std::string err;
    if (!io::read_fasta_file("tests/data/mtdna_human.fa", &hrec, &err)) {
        printf("  cannot read human mtDNA: %s\n", err.c_str());
        printf("  (run from the repository root)\n");
        return 2;
    }
    if (!io::read_fasta_file("tests/data/mtdna_chimp.fa", &crec, &err)) {
        printf("  cannot read chimp mtDNA: %s\n", err.c_str());
        return 2;
    }
    const std::string hum = hrec[0].sequence;
    const std::string chi = crec[0].sequence;
    printf("  human mtDNA : %s, %zu bp\n", hrec[0].id.c_str(), hum.size());
    printf("  chimp mtDNA : %s, %zu bp\n", crec[0].id.c_str(), chi.size());
    if (hum.size() < 10000 || chi.size() < 10000) {
        printf("  FAIL: data files look truncated\n");
        return 1;
    }

    if (!device_available()) {
        // The whole point of B2 is pairwise correctness on real input, which needs a
        // device. Report plainly and exit 77 (skipped), as the API test does: a
        // device-less host is not a failing library.
        printf("\n  SKIP: no device visible; real-sequence correctness NOT verified here.\n");
        printf("RESULT: SKIP -- needs a GPU (exit 77)\n");
        return 77;
    }

    // ---- 1. short REAL fragments, exact matches and mismatches -----------
    printf("\n-- 1. real fragments, low divergence (same molecule) --\n");
    {
        std::vector<std::string> pats, texts;
        // tRNA-rich region (around 4500-6500), a GC-rich and structured stretch
        const size_t starts[] = {1000, 3000, 4500, 6000, 9000, 12000, 15000};
        for (size_t s : starts) {
            const std::string frag = slice(hum, s, 400);
            if (frag.size() < 400) continue;
            pats.push_back(frag);
            texts.push_back(frag);                       // identical
            pats.push_back(frag);
            std::string mut = frag;                      // 2 substitutions
            mut[10] = (mut[10] == 'A' ? 'C' : 'A');
            mut[200] = (mut[200] == 'G' ? 'T' : 'G');
            texts.push_back(mut);
        }
        compare_all(pats, texts, 64, "identical + 2 substitutions");
    }

    // ---- 2. the D-loop: the repeat-heavy region --------------------------
    printf("\n-- 2. D-loop / control region (repeat-rich, the hard case) --\n");
    {
        std::vector<std::string> pats, texts;
        // Human mtDNA control region is ~16024-576. Take windows across the origin.
        const std::string dloop = slice(hum, 16000, 569) + slice(hum, 0, 600);
        pats.push_back(dloop); texts.push_back(dloop);       // self
        std::string shifted = dloop.substr(3) + dloop.substr(0, 3);   // small shift
        pats.push_back(dloop); texts.push_back(shifted);
        std::string mut = dloop;
        for (size_t i = 0; i + 7 < mut.size(); i += 23) mut[i] = (mut[i] == 'A' ? 'C' : 'A');
        pats.push_back(dloop); texts.push_back(mut);
        compare_all(pats, texts, 64, "D-loop windows");
    }

    // ---- 3. REAL divergent homolog: human vs chimp mtDNA -----------------
    printf("\n-- 3. human vs chimp mtDNA (real divergence, ~88%% identity) --\n");
    {
        std::vector<std::string> pats, texts;
        // Co-linear windows; genome sizes differ by 15 bp, so no window aligns cleanly.
        const size_t wins[] = {500, 2000, 4000, 8000, 11000, 14000};
        for (size_t w : wins) {
            const std::string p = slice(hum, w, 600);
            const std::string t = slice(chi, w, 600);
            if (p.size() < 600 || t.size() < 600) continue;
            pats.push_back(p); texts.push_back(t);
        }
        // 600 bp of ~88% identity is ~72 edits: above smax=64, so these MUST be
        // unresolved, and that is the point -- the test asserts the API SAYS SO
        // rather than returning a truncated alignment as if it were complete.
        compare_all(pats, texts, 64, "600 bp divergent, smax=64");
        printf("        (unresolved above is EXPECTED: d ~72 > smax=64)\n");

        // With a bound that can hold the distance, the same pairs must resolve.
        compare_all(pats, texts, 200, "600 bp divergent, smax=200");
    }

    // ---- 4. full-length real sequences, the honest stress case -----------
    printf("\n-- 4. full-length molecules (16.5 kb) --\n");
    {
        std::vector<std::string> pats{hum, hum}, texts{hum, chi};
        // smax must cover ~12%% divergence over 16.5 kb (~2000 edits).
        compare_all(pats, texts, 511, "whole mtDNA, smax=511");
        printf("        (the human-vs-chimp case needs d > 511; unresolved is\n");
        printf("         expected and correct -- see the count above)\n");
    }

    // ---- 5. the N and lowercase handling on real data --------------------
    printf("\n-- 5. real ambiguity character (the N in human mtDNA) --\n");
    {
        const size_t npos = hum.find('N');
        if (npos == std::string::npos) {
            printf("  NOTE: no N found; skipping (data changed?)\n");
        } else {
            printf("  N at position %zu; aligning a window around it\n", npos);
            const size_t from = (npos > 100) ? npos - 100 : 0;
            const std::string p = slice(hum, from, 200);
            std::vector<std::string> pats{p, p}, texts{p, std::string(p.rbegin(), p.rend())};
            compare_all(pats, texts, 200, "window containing N (+ its reverse)");
        }
    }

    printf("\n");
    if (!g_ran) { printf("RESULT: FAIL -- no assertions executed\n"); return 1; }
    if (g_fail == 0) { printf("RESULT: PASS -- real-sequence alignment verified\n"); return 0; }
    printf("RESULT: FAIL (%d)\n", g_fail);
    return 1;
}
