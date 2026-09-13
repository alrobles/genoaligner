// genoaligner — B4: batch smax semantics, pinned.
//
// THE CONTRACT (api.hpp): a batch uses ONE smax -- the MINIMUM of the requests' --
// because the kernel takes a single bound. A per-request smax is a limit the CALLER
// set, so silently granting more would resolve pairs they expected abandoned; taking
// the minimum can only under-serve, which is visible via resolved_count.
//
// WHY A TEST, NOT A COMMENT
// -------------------------
// That paragraph was in the implementation and in the header while nothing checked
// it. A contract that no test exercises is a description of what the author believed
// at the time -- and this project has already shipped one wrong claim of exactly that
// kind (a distance estimate of ~72 where the measured value was ~300).
//
// The test is built so that the two candidate behaviours give DIFFERENT observable
// answers, which is what makes it a test rather than a restatement:
//
//   request A: 300 bp window, smax=200   (distance ~30, would resolve at 200)
//   request B: 300 bp window, smax=8     (distance ~30, CANNOT resolve at 8)
//
//   if the batch uses MIN (=8)  -> BOTH unresolved, resolved_count = 0
//   if the batch uses MAX (=200)-> BOTH resolved,   resolved_count = 2
//
// So the assertion "resolved_count == 0" is exactly the claim "the batch used the
// minimum", and it fails loudly if anyone changes the implementation to take the
// maximum. It also documents the COST honestly: the narrow request governed, and
// request A -- which would have resolved -- did not.

#include "include/genoaligner/api.hpp"
#include "include/genoaligner/io/fasta.hpp"
#include "src/reference/edit_distance_cpu.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace genoaligner;

static int g_fail = 0;
static void check(bool ok, const char* what)
{
    if (!ok) { printf("  FAIL: %s\n", what); ++g_fail; }
}

static std::string slice(const std::string& s, size_t from, size_t len)
{
    if (from >= s.size()) return std::string();
    return s.substr(from, std::min(len, s.size() - from));
}

int main()
{
    printf("genoaligner — batch smax semantics (B4)\n");
    printf("  backend : %s\n  device  : %s\n\n", backend_name(), device_name());

    std::vector<io::FastaRecord> hrec;
    std::string err;
    if (!io::read_fasta_file("tests/data/mtdna_human.fa", &hrec, &err)) {
        printf("  cannot read tests/data/mtdna_human.fa: %s\n", err.c_str());
        printf("  (run from the repository root)\n");
        return 2;
    }
    const std::string hum = hrec[0].sequence;

    if (!device_available()) {
        printf("  SKIP: no device visible; batch semantics NOT verified here.\n");
        printf("RESULT: SKIP -- needs a GPU (exit 77)\n");
        return 77;
    }

    // Two windows from the same molecule; distances MEASURED, not assumed.
    // A mutation every 15 bases over 300 bp gives d=20 (checked with the CPU DP
    // below): comfortably over the narrow bound of 8, comfortably under 200. The
    // first attempt mutated every 40 bases, which gave exactly d=8 -- right at the
    // boundary, so the narrow bound would NOT have bound and the test would have
    // failed for a reason unrelated to what it is testing.
    const std::string w1 = slice(hum, 1000, 300);
    const std::string w2 = slice(hum, 5000, 300);
    std::string m1 = w1, m2 = w2;
    for (size_t i = 0; i + 5 < m1.size(); i += 15) m1[i] = (m1[i] == 'A') ? 'C' : 'A';
    for (size_t i = 0; i + 5 < m2.size(); i += 15) m2[i] = (m2[i] == 'A') ? 'C' : 'A';

    const int d1 = edit_distance_cpu(w1.data(), (int)w1.size(), m1.data(), (int)m1.size());
    const int d2 = edit_distance_cpu(w2.data(), (int)w2.size(), m2.data(), (int)m2.size());
    printf("  measured distances: d1=%d  d2=%d  (need 8 < d <= 200 for both)\n", d1, d2);
    printf("  (that range is what makes the narrow bound bind and the wide one\n");
    printf("   resolve, which is what distinguishes min-semantics from max-semantics)\n");

    if (d1 > 200 || d2 > 200) {
        printf("  !!! distances too large for this test to be meaningful; adjust windows\n");
        return 1;
    }
    if (d1 <= 8 || d2 <= 8) {
        printf("  !!! distances too small: the narrow bound would not bind; adjust windows\n");
        return 1;
    }

    // ---- mixed bounds: the narrow request must govern ---------------------
    printf("\n-- batch with MIXED bounds (smax=200 and smax=8) --\n");
    {
        std::vector<AlignRequest> reqs(2);
        reqs[0].pattern = w1.data(); reqs[0].pattern_len = (int)w1.size();
        reqs[0].text    = m1.data(); reqs[0].text_len    = (int)m1.size();
        reqs[0].smax    = 200;
        reqs[1].pattern = w2.data(); reqs[1].pattern_len = (int)w2.size();
        reqs[1].text    = m2.data(); reqs[1].text_len    = (int)m2.size();
        reqs[1].smax    = 8;

        BatchResult b = align_batch(reqs);
        check(b.ok(), "mixed-bounds batch is accepted (both bounds are legal)");
        printf("  resolved=%d unresolved=%d   (MIN semantics predicts 0 and 2)\n",
               b.resolved_count, b.unresolved_count);

        // THE assertion: min semantics, not max.
        check(b.resolved_count == 0,
              "the batch used the MINIMUM smax (a max would have resolved both)");
        check(b.unresolved_count == 2, "both requests unresolved under the narrow bound");

        // And the cost is real and observable: request A alone WOULD have resolved.
        // Score-only: the trace kernel's per-block shared workspace is
        // (smax+1)(2*smax+3) ints -- past the 64 KiB device limit beyond
        // smax ~88, so the API refuses with_cigar at smax=200 by design. What
        // is asserted here is resolution under the wide bound, which the score
        // kernel answers.
        std::vector<AlignRequest> alone{reqs[0]};
        alone[0].with_cigar = false;
        BatchResult bA = align_batch(alone);
        printf("  the same request alone (smax=200): resolved=%d  <- shows what was lost\n",
               bA.resolved_count);
        check(bA.resolved_count == 1, "the wide request resolves on its own");
        check(bA.results[0].score == d1, "and its score matches the CPU DP");
    }

    // ---- the supported workaround: group by bound ------------------------
    printf("\n-- grouping by bound, as the header recommends --\n");
    {
        std::vector<AlignRequest> wide(2);
        for (int i = 0; i < 2; ++i) {
            const std::string& p = (i == 0) ? w1 : w2;
            const std::string& q = (i == 0) ? m1 : m2;
            wide[(size_t)i].pattern = p.data(); wide[(size_t)i].pattern_len = (int)p.size();
            wide[(size_t)i].text    = q.data(); wide[(size_t)i].text_len    = (int)q.size();
            wide[(size_t)i].smax    = 200;
            wide[(size_t)i].with_cigar = false;   // see note above: smax>~88 is
                                                  // score-only by device limit
        }
        BatchResult b = align_batch(wide);
        printf("  both with smax=200: resolved=%d unresolved=%d\n",
               b.resolved_count, b.unresolved_count);
        check(b.resolved_count == 2, "grouping by bound recovers both resolutions");
        check(b.results[0].score == d1 && b.results[1].score == d2,
              "and both scores match the CPU DP");
    }

    printf("\n");
    if (g_fail == 0) {
        printf("RESULT: PASS -- batch uses the minimum smax, as documented\n");
        return 0;
    }
    printf("RESULT: FAIL (%d)\n", g_fail);
    return 1;
}
