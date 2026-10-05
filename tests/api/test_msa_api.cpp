// genoaligner — public MSA API test.
//
// WHAT THIS VERIFIES
// ------------------
// Unlike the pairwise kernels, the MSA engine is pure host code: there is no
// shim path and no device dependency, so EVERY build asserts real results --
// output shape, codon invariants, QC fields, validation refusals and
// determinism -- and the documented header example is executed verbatim so
// the documented usage cannot drift from the working usage.
//
// Parity vs the engine itself is checked by calling genomsa::msa_align
// directly on the same inputs: the public wrapper must add no semantic
// difference in dna mode.

#include "genoaligner/api.hpp"
#include "genoaligner/msa/msa.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace genoaligner;

static int g_fail = 0;
static void check(bool ok, const char* what)
{
    if (!ok) { printf("  FAIL: %s\n", what); ++g_fail; }
}

static bool rectangular(const std::vector<std::string>& rows)
{
    if (rows.empty()) return true;
    const size_t w = rows[0].size();
    for (const auto& r : rows) if (r.size() != w) return false;
    return true;
}

// Gaps may only appear in whole-codon runs: every maximal run of '-' in a row
// must have length %3 and start at a multiple of 3.
static bool codon_gap_blocks(const std::string& row)
{
    for (size_t i = 0; i < row.size();) {
        if (row[i] != '-') { ++i; continue; }
        const size_t s = i;
        while (i < row.size() && row[i] == '-') ++i;
        if ((i - s) % 3 != 0 || s % 3 != 0) return false;
    }
    return true;
}

// Ungapping a row must reproduce the input sequence exactly.
static std::string ungap(const std::string& row)
{
    std::string s;
    for (char c : row) if (c != '-') s += c;
    return s;
}

int main()
{
    printf("genoaligner public MSA API test (host engine, real results)\n");

    // ------------------------------------------------------------ example
    // The api.hpp documented example, verbatim.
    {
        MsaRequest req;
        req.mode = MsaMode::codon;
        req.gc_def = 2;                    // vertebrate mitochondrial
        req.seqs = {"ATGATAATCACC", "ATGATTATCACCTGA"};
        MsaResult r = msa_align(req);
        check(r.ok(), "header example resolves");
        check(r.aligned.size() == 2, "header example: 2 rows");
        check(rectangular(r.aligned), "header example: rectangular");
        check(r.width == (r.aligned.empty() ? 0 : (int)r.aligned[0].size()),
              "header example: width field");
        check(r.qc.size() == 2, "header example: qc per input seq");
        check(r.qc[0].frame >= 0 && r.qc[0].frame <= 2,
              "header example: frame in [0,2]");
        check(r.qc[0].stops >= 0, "header example: stops reported");
    }

    // --------------------------------------------------------- dna mode
    {
        MsaRequest req;
        req.mode = MsaMode::dna;
        req.seqs = {"ACGTACGTACGT", "ACGTACGTTCGT",
                    "ACGTTCGTACGT", "ACGTACGTACGTA"};
        MsaResult r = msa_align(req);
        check(r.ok(), "dna: resolves");
        check(r.aligned.size() == 4, "dna: row count");
        check(rectangular(r.aligned), "dna: rectangular");
        check(r.width >= 13, "dna: width >= longest input");
        for (size_t i = 0; i < 4; ++i)
            check(ungap(r.aligned[i]) == req.seqs[i],
                  "dna: ungapped row reproduces input (order kept)");
        for (const auto& q : r.qc)
            check(q.frame == -1 && q.stops == -1 && q.partial == -1,
                  "dna: qc fields are -1 outside codon mode");
        // Determinism is part of the engine spec: identical input twice.
        MsaResult r2 = msa_align(req);
        check(r2.aligned == r.aligned, "dna: deterministic on repeat call");
        // Parity: the wrapper adds no semantic difference vs the engine.
        genomsa::Params P;
        std::vector<std::string> direct = genomsa::msa_align(req.seqs, P);
        check(direct == r.aligned, "dna: identical to direct engine call");
    }

    // ----------------------------------------------------- protein mode
    {
        MsaRequest req;
        req.mode = MsaMode::protein;
        req.seqs = {"MKTAYIAKQRQISFVK", "MKTAYIAKQRQISFVKE",
                    "MKTAZIAKQRQISFVK"};    // Z = E|Q ambiguity set
        MsaResult r = msa_align(req);
        check(r.ok(), "protein: resolves");
        check(rectangular(r.aligned), "protein: rectangular");
        check(r.aligned.size() == 3, "protein: row count");
    }

    // -------------------------------------------------------- codon mode
    // Whole-codon invariants hold for the strict configuration
    // (local_frame=false, refine=0): output %3, gaps only in codon blocks.
    {
        MsaRequest req;
        req.mode = MsaMode::codon;
        req.gc_def = 1;
        req.codon_local_frame = false;
        req.codon_refine = 0;
        req.seqs = {"ATGAAACCCGGGTTT",          // 5 codons
                    "ATGAAAGGGTTT",             // 4 codons (1-codon deletion)
                    "ATGAAACCCGGGTTTAAA"};      // 6 codons (1 extra codon)
        MsaResult r = msa_align(req);
        check(r.ok(), "codon strict: resolves");
        check(rectangular(r.aligned), "codon strict: rectangular");
        check(r.width % 3 == 0, "codon strict: width multiple of 3");
        for (const auto& row : r.aligned)
            check(codon_gap_blocks(row),
                  "codon strict: indels are whole-codon blocks");
        for (size_t i = 0; i < 3; ++i)
            check(ungap(r.aligned[i]) == req.seqs[i],
                  "codon strict: ungapped row reproduces input");
        check(r.qc.size() == 3, "codon strict: qc per seq");
        check(r.qc[0].stops == 0, "codon strict: clean CDS has no stops");
        check(r.qc[2].frame == 0 && r.qc[2].stops == 0,
              "codon strict: clean extra codon keeps frame 0");
    }

    // QC stop counting: codon_encode picks the frame with FEWEST stops, so a
    // stop only survives when every frame carries one. "TAAATAAGTAG" reads
    // TAA|ATA|AGT|AG(partial) in frame 0, AAT|TAA|GTA in frame 1 and
    // ATA|AGT|TAG in frame 2 -- min stops = 1, tie broken to frame 0. Its
    // trailing partial decodes as NNN: partial codons are reported in QC,
    // not silently dropped.
    {
        MsaRequest req;
        req.mode = MsaMode::codon;
        req.codon_local_frame = false;
        req.seqs = {"TAAATAAGTAG", "ATGAAAGGGTTT"};
        MsaResult r = msa_align(req);
        check(r.ok(), "codon qc: resolves");
        check(r.qc[0].frame == 0 && r.qc[0].stops == 1 && r.qc[0].partial == 1,
              "codon qc: frame/stops/partial reported");
        check(ungap(r.aligned[0]).size() >= req.seqs[0].size(),
              "codon qc: partial codon decodes to NNN");
        check(r.qc[1].stops == 0, "codon qc: clean CDS has none");
    }

    // The default codon path (local_frame on -> refine >= 1) is not required
    // to be %3; it must still be rectangular and input-faithful.
    {
        MsaRequest req;
        req.mode = MsaMode::codon;
        req.seqs = {"ATGAAACCCGGGTTT", "ATGAAAGGGTTT", "ATGAAACCCGGGTTTAAA"};
        MsaResult r = msa_align(req);
        check(r.ok(), "codon default: resolves");
        check(rectangular(r.aligned), "codon default: rectangular");
        for (size_t i = 0; i < 3; ++i)
            check(ungap(r.aligned[i]) == req.seqs[i],
                  "codon default: ungapped row reproduces input");
    }

    // ----------------------------------------------------------- degenerate
    {
        MsaRequest req;
        req.seqs = {"ACGTACGT"};
        MsaResult r = msa_align(req);
        check(r.ok() && r.aligned.size() == 1 && r.aligned[0] == req.seqs[0],
              "one sequence returns itself");
    }

    // --------------------------------------------------------- validation
    {
        MsaRequest req;
        check(msa_align(req).status == MsaResult::Status::invalid_argument,
              "validation: empty request refused");
        req.seqs = {"ACGT", ""};
        check(msa_align(req).status == MsaResult::Status::invalid_argument,
              "validation: empty sequence refused");
        req.seqs = {"ACGT", "ACGT!"};
        check(msa_align(req).status == MsaResult::Status::invalid_argument,
              "validation: bad character refused");
        req.seqs = {"ACGT", "AC-T"};
        check(msa_align(req).status == MsaResult::Status::invalid_argument,
              "validation: pre-gapped input refused");
        req.seqs = {"ACGT", "AGGT"};
        req.codon_refine = 2;
        check(msa_align(req).status == MsaResult::Status::invalid_argument,
              "validation: codon_refine outside codon mode refused");
        req.codon_refine = -1;
        req.mode = MsaMode::codon;
        check(msa_align(req).status == MsaResult::Status::invalid_argument,
              "validation: negative codon_refine refused");
        req.codon_refine = 0;
        req.gc_def = 3;                 // still codon mode: table 3 unsupported
        check(msa_align(req).status == MsaResult::Status::invalid_argument,
              "validation: unsupported gc_def refused");
        req.gc_def = 2;
        req.mode = MsaMode::dna;        // gc 2 means nothing in dna mode
        check(msa_align(req).status == MsaResult::Status::invalid_argument,
              "validation: gc_def outside codon mode refused");
        req.gc_def = 1;
        req.mode = static_cast<MsaMode>(99);
        check(msa_align(req).status == MsaResult::Status::invalid_argument,
              "validation: casted-out-of-range mode refused");
    }

    // ------------------------------------------------------------- report
    if (g_fail == 0) printf("MSA API test: PASS\n");
    else             printf("MSA API test: %d FAILURE(S)\n", g_fail);
    return g_fail ? 1 : 0;
}
