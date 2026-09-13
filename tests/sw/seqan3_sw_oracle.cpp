// genoaligner — Fase B external oracle: SeqAn3 LOCAL alignment (Smith-Waterman).
//
// Same role as seqan_probe/seqan3_oracle.cpp played for WFA edit distance: a
// third implementation we did not write, so the gate is not self-consistency.
// This one runs align_pairwise under method_local with an affine gap scheme.
//
// SCORING SCHEME MAPPING -- the detail that must be pinned, not assumed:
//   ours:    gap of length L costs gap_open + gap_extend * (L-1)
//            (E[i][j] = H[i][j-1] - gap_open opens at length 1)
//   seqan3:  gap of length L costs open_score + extension_score * L
//            ("the score for opening a gap is the sum of the gap score and the
//             gap open score" -- align_config_gap_cost_affine.hpp)
//   => seqan3 open_score = -(gap_open - gap_extend), extension_score = -gap_extend
// The --selftest below verifies this on cases where the two conventions would
// differ (a length-2 gap costs go+ge under ours = open+2*ext under theirs).
//
// SCHEME UNDER TEST: match=1, mismatch=-1, gap_open=2, gap_extend=1
//   -> seqan3: match_score{1}, mismatch_score{-1}, open{-1}, extension{-1}
// (The emitted TSV from test_sw_trace uses exactly this scheme.)
//
// INPUT:  the TSV emitted by test_sw_trace(.cpp) / the GPU harness:
//           idx \t text \t pattern \t score \t start_i \t start_j \t end_i \t end_j \t cigar
// OUTPUT: same rows + seqan3_score \t s1_begin \t s1_end \t s2_begin \t s2_end
//   so a downstream check can compare scores AND the aligned span. Positions may
//   legitimately differ on ties; the SCORE must not.
//
// BUILD (cluster toolchain; SeqAn3 3.4 needs g++ >= 12, C++23):
//   /kuhpc/sw/gcc/14.2/bin/g++ -std=c++23 -w -O2 \
//       -I <seqan3>/include -o seqan3_sw_oracle seqan3_sw_oracle.cpp
//   run with LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <seqan3/alphabet/nucleotide/dna4.hpp>
#include <seqan3/alignment/configuration/align_config_gap_cost_affine.hpp>
#include <seqan3/alignment/configuration/align_config_method.hpp>
#include <seqan3/alignment/configuration/align_config_output.hpp>
#include <seqan3/alignment/configuration/align_config_scoring_scheme.hpp>
#include <seqan3/alignment/pairwise/align_pairwise.hpp>
#include <seqan3/alignment/scoring/nucleotide_scoring_scheme.hpp>

namespace {

constexpr int K_MATCH = 1, K_MISMATCH = -1, K_GO = 2, K_GE = 1;

std::vector<seqan3::dna4> to_dna4(const std::string &s)
{
    std::vector<seqan3::dna4> out;
    out.reserve(s.size());
    for (char c : s) {
        seqan3::dna4 letter{};
        letter.assign_char(c);
        out.push_back(letter);
    }
    return out;
}

struct OracleOut {
    int score;
    int s1_begin, s1_end, s2_begin, s2_end;   // seqan3 positions (end-exclusive)
};

OracleOut seqan3_sw(const std::string &text, const std::string &pattern)
{
    OracleOut o{0, -1, -1, -1, -1};
    if (text.empty() || pattern.empty()) return o;

    auto tv = to_dna4(text);     // sequence1 == our row axis (text)
    auto pv = to_dna4(pattern);  // sequence2 == our column axis (pattern)

    auto cfg = seqan3::align_cfg::method_local{}
             | seqan3::align_cfg::scoring_scheme{
                   seqan3::nucleotide_scoring_scheme{seqan3::match_score{K_MATCH},
                                                     seqan3::mismatch_score{K_MISMATCH}}}
             | seqan3::align_cfg::gap_cost_affine{
                   seqan3::align_cfg::open_score{-(K_GO - K_GE)},
                   seqan3::align_cfg::extension_score{-K_GE}}
             | seqan3::align_cfg::output_score{}
             | seqan3::align_cfg::output_begin_position{}
             | seqan3::align_cfg::output_end_position{};

    for (auto &&res : seqan3::align_pairwise(std::pair{tv, pv}, cfg)) {
        o.score    = res.score();
        o.s1_begin = (int)res.sequence1_begin_position();
        o.s1_end   = (int)res.sequence1_end_position();
        o.s2_begin = (int)res.sequence2_begin_position();
        o.s2_end   = (int)res.sequence2_end_position();
    }
    return o;
}

std::vector<std::string> split_tab(const std::string &line)
{
    std::vector<std::string> parts;
    std::string cur;
    std::istringstream ss(line);
    while (std::getline(ss, cur, '\t')) parts.push_back(cur);
    return parts;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc == 2 && std::string(argv[1]) == "--selftest") {
        // Hand-derived expectations under OUR scheme (match=1, mismatch=-1,
        // go=2, ge=1), cross-checked against tests/sw/test_sw_reference.cpp.
        // The length-2-gap case is the one that pins the open/extension mapping:
        // ours costs go + ge*(L-1) = 3; a wrong mapping to open=-go would give 4.
        struct Case { const char *t, *p; int want; };
        const std::vector<Case> cases = {
            {"ACGT", "ACGT", 4},
            {"A", "A", 1},
            {"AAAAAAAA", "CCCCCCCC", 0},           // no positive alignment
            {"ACGTTTACGT", "ACGTACGT", 5},         // 8 - (2+1) : THE mapping pin
            {"ACGTACGT", "ACGTTTACGT", 5},         // same, other axis
            {"ACGTACGT", "TGCATGCA", 1},           // shares one 'A'
            {"ACACACAC", "ACAC", 4},
        };
        int bad = 0;
        std::printf("SeqAn3 SW oracle self-test (local, affine go=2 ge=1)\n\n");
        for (const auto &c : cases) {
            const int got = seqan3_sw(c.t, c.p).score;
            const bool ok = (got == c.want);
            bad += !ok;
            std::printf("  %-12s vs %-12s seqan3=%3d want=%3d  %s\n",
                        c.t, c.p, got, c.want, ok ? "ok" : "** MISMATCH **");
        }
        std::printf("\nRESULT: %s\n", bad ? "MISMATCHES" : "ALL OK");
        return bad ? 1 : 0;
    }

    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <emit.tsv> <out.tsv>\n"
            "       %s --selftest\n", argv[0], argv[0]);
        return 2;
    }

    std::ifstream in(argv[1]);
    if (!in) { std::fprintf(stderr, "ERROR: cannot open %s\n", argv[1]); return 2; }
    std::ofstream out(argv[2]);
    if (!out) { std::fprintf(stderr, "ERROR: cannot open %s\n", argv[2]); return 2; }

    std::string line;
    size_t n = 0;
    int disagreements = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto f = split_tab(line);
        if (f[0] == "idx") {           // header: append the oracle columns
            out << line << "\tseqan3_score\ts1_begin\ts1_end\ts2_begin\ts2_end\n";
            continue;
        }
        if (f.size() < 9) { std::fprintf(stderr, "WARN: short row skipped: %s\n", line.c_str()); continue; }
        const OracleOut o = seqan3_sw(f[1], f[2]);
        const int ours = std::atoi(f[3].c_str());
        if (o.score != ours) ++disagreements;
        out << line << '\t' << o.score << '\t' << o.s1_begin << '\t' << o.s1_end
            << '\t' << o.s2_begin << '\t' << o.s2_end << '\n';
        ++n;
    }
    std::fprintf(stderr, "%s: %zu rows, %d score disagreements\n",
                 argv[0], n, disagreements);
    return disagreements ? 1 : 0;
}
