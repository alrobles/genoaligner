// genoaligner Fase 3 — SeqAn3 oracle (third independent implementation).
//
// WHY A C++ ORACLE AND NOT PYTHON:
//   edlib and rapidfuzz are C libraries with Python bindings, so the Python
//   oracle covers them. SeqAn3 is a header-only C++23 library with no Python
//   binding, so this is a small standalone tool instead.
//
// WHAT IT DOES:
//   Reads the harness's --emit TSV
//       index <TAB> pattern <TAB> text <TAB> gpu <TAB> cpu <TAB> label
//   and writes a 4th column with the SeqAn3 score, so all three oracles can be
//   compared in one pass:
//       index <TAB> pattern <TAB> text <TAB> gpu <TAB> cpu <TAB> seqan3 <TAB> label
//
// SIGN CONVENTION (cost one debugging cycle):
//   SeqAn3 maximises, so `edit_scheme` scores match 0, mismatch -1, gap -1.
//   The edit DISTANCE is therefore the NEGATION of SeqAn3's score. Verified on
//   the R1 minimal reproducers (GG/TTT -> -3, C/GT -> -2).
//
// BUILD (every flag below was needed; see the comments for why):
//   /kuhpc/sw/gcc/14.2/bin/g++ -std=c++23 -w -O2 \
//       -I <seqan3>/include -o seqan3_oracle seqan3_oracle.cpp
//   Run with LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64 (GLIBCXX_3.4.32).
//
//   - g++ 14.2 is REQUIRED: SeqAn3 3.4 needs GCC >= 12 and C++23, and the
//     cluster's default g++ is 11.5. The compiler/gcc/14.2 module does NOT put
//     g++ on PATH, so the absolute path is used.
//   - -w silences a large volume of SeqAn3 internal warnings; there are no
//     errors without it, but the noise buries the real diagnostics.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <seqan3/alphabet/nucleotide/dna4.hpp>
#include <seqan3/alignment/configuration/align_config_edit.hpp>
#include <seqan3/alignment/configuration/align_config_method.hpp>
#include <seqan3/alignment/pairwise/align_pairwise.hpp>

// Convert an ACGT string to SeqAn3 dna4.
static std::vector<seqan3::dna4> to_dna4(const std::string &s)
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

// Unit-cost edit distance via SeqAn3. Returns -1 on an unparsable sequence.
static int seqan_edit_distance(const std::string &pattern, const std::string &text)
{
    auto pv = to_dna4(pattern);
    auto tv = to_dna4(text);

    // The config is a PIPE of method_global and edit_scheme. Omitting
    // method_global trips a static_assert deep inside alignment_configurator.
    auto config = seqan3::align_cfg::method_global{}
                | seqan3::align_cfg::edit_scheme;

    int score = 0;
    for (auto &&res : seqan3::align_pairwise(std::pair{pv, tv}, config))
        score = res.score();

    // SeqAn3 maximises; edit distance is the negation.
    return -score;
}

static std::vector<std::string> split_tab(const std::string &line)
{
    std::vector<std::string> parts;
    std::string cur;
    std::istringstream ss(line);
    while (std::getline(ss, cur, '\t')) parts.push_back(cur);
    return parts;
}

int main(int argc, char **argv)
{
    if (argc == 2 && std::string(argv[1]) == "--selftest") {
        // Prove the oracle against hand-computed unit-cost edit distances before
        // trusting it as a reference. Includes the R1 minimal reproducers.
        struct Case { const char *p; const char *t; int want; };
        const std::vector<Case> cases = {
            {"ACGT", "ACGT", 0}, {"ACGT", "ACGA", 1},
            {"GG", "TTT", 3},            // R1: free insertions
            {"C", "GT", 2},              // R1: direction confusion
            {"CT", "CCC", 2},            // R1
            {"CG", "GAC", 3},            // R1
            {"A", "AA", 1},              // R1: sentinel corruption
            {"AAAA", "AAAAA", 1},
            {"TTTTTTTT", "TTTT", 4},
            {"ACGTACGT", "TTTTTTTT", 6},
            {"", "", 0},
            {"ACGT", "", 4},
            {"", "ACGT", 4},
        };
        int bad = 0;
        std::printf("SeqAn3 oracle self-test (unit-cost edit distance)\n\n");
        for (const auto &c : cases) {
            const int got = seqan_edit_distance(c.p, c.t);
            const bool ok = (got == c.want);
            bad += !ok;
            std::printf("  %-10s vs %-10s seqan3=%3d want=%3d  %s\n",
                        c.p[0] ? c.p : "(empty)", c.t[0] ? c.t : "(empty)",
                        got, c.want, ok ? "ok" : "** MISMATCH **");
        }
        std::printf("\nRESULT: %s\n", bad ? "MISMATCHES" : "ALL OK");
        return bad ? 1 : 0;
    }

    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <emit.tsv> <out.tsv>\n"
            "       %s --selftest\n"
            "  reads the harness --emit output and adds a SeqAn3 column\n", argv[0], argv[0]);
        return 2;
    }
    const std::string in_path = argv[1], out_path = argv[2];

    std::ifstream in(in_path);
    if (!in) { std::fprintf(stderr, "ERROR: cannot open %s\n", in_path.c_str()); return 2; }
    std::ofstream out(out_path);
    if (!out) { std::fprintf(stderr, "ERROR: cannot open %s\n", out_path.c_str()); return 2; }

    std::string line;
    size_t n = 0;
    int bad = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') { continue; }
        auto p = split_tab(line);

        if (p[0] == "index") {
            // Header: insert seqan3 before the trailing label column.
            out << "index\tpattern\ttext\tgpu\tcpu\tseqan3\tlabel\n";
            continue;
        }
        if (p.size() < 4) { continue; }

        const std::string &pattern = p[1], &text = p[2];
        const int gpu = std::atoi(p[3].c_str());

        // Skipped/abandoned cases (isD beyond smax) carry gpu < 0 and are not
        // scored; emit -1 so the column stays aligned.
        if (gpu < 0) {
            out << p[0] << '\t' << pattern << '\t' << text << '\t'
                << p[3] << '\t' << (p.size() > 4 ? p[4] : "-1") << '\t'
                << "-1\t" << (p.size() > 5 ? p[5] : "") << '\n';
            continue;
        }

        const int d = seqan_edit_distance(pattern, text);
        if (d < 0) ++bad;

        out << p[0] << '\t' << pattern << '\t' << text << '\t'
            << p[3] << '\t' << (p.size() > 4 ? p[4] : "-1") << '\t'
            << d << '\t' << (p.size() > 5 ? p[5] : "") << '\n';
        ++n;
    }

    std::fprintf(stderr, "seqan3 oracle: scored %zu cases%s\n",
                 n, bad ? " (some unparsable)" : "");
    return bad ? 1 : 0;
}
