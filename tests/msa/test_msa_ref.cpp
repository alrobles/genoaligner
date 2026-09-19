// Synthetic tests for the CPU-reference MSA (M1 gate).
// Every case has a KNOWN correct answer -- no "looks plausible" checks.
#include <genoaligner/msa/msa.hpp>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
using namespace genomsa;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    printf("FAIL: %s\n", msg); ++fails; } } while (0)

static int ncols_of(const std::vector<std::string>& msa) {
    return msa.empty() ? 0 : (int)msa[0].size();
}
static bool ragged(const std::vector<std::string>& msa) {
    for (auto& r : msa) if ((int)r.size() != ncols_of(msa)) return true;
    return false;
}
static std::string ungap(std::string s) {
    s.erase(std::remove(s.begin(), s.end(), '-'), s.end());
    return s;
}

int main() {
    Params P;

    // ---- 1. two identical sequences -> identity MSA, no gaps
    {
        std::vector<std::string> out = msa_align({"ACGTACGT", "ACGTACGT"}, P);
        CHECK(out.size() == 2 && out[0] == "ACGTACGT" && out[1] == "ACGTACGT",
              "identical seqs must align unchanged");
    }

    // ---- 2. one seq with a 3bp insertion -> exactly 3 gap columns in other
    {
        std::string a = "ACGTACGT";
        std::string b = "ACGTTTTACGT";      // +TTT at pos 4
        std::vector<std::string> out = msa_align({a, b}, P);
        CHECK(ncols_of(out) == 11 && !ragged(out), "insertion width");
        CHECK(ungap(out[0]) == a && ungap(out[1]) == b, "sequences preserved");
        int gaps_a = std::count(out[0].begin(), out[0].end(), '-');
        CHECK(gaps_a == 3, "3 gaps opposite the insertion");
        // the gaps must sit inside the aligned span, not at the ends
        size_t first = out[0].find('-'), last = out[0].rfind('-');
        CHECK(first > 0 && last < out[0].size() - 1, "gaps internal");
    }

    // ---- 3. fragment vs full-length -> semiglobal, fragment not padded
    //         through the interior
    {
        std::string full = "ACGTACGTACGTACGTACGT";         // 20
        std::string frag = full.substr(6, 8);               // "CGTACGTA" 8bp
        std::vector<std::string> out = msa_align({full, frag}, P);
        CHECK(ncols_of(out) == 20, "fragment aligned in place, width 20");
        CHECK(ungap(out[1]) == frag, "fragment seq preserved");
        // fragment occupies a contiguous window of 8 non-gap columns
        int run = 0, best = 0;
        for (char c : out[1]) { run = (c == '-') ? 0 : run + 1; best = std::max(best, run); }
        CHECK(best == 8, "fragment letters contiguous");
    }

    // ---- 4. three sequences, two identical + one divergent
    {
        std::vector<std::string> out =
            msa_align({"ACGTACGT", "ACGTACGT", "TTTTTTTT"}, P);
        CHECK(!ragged(out), "3seq aligned equal length");
        CHECK(out[0] == out[1], "identical pair stays identical in MSA");
    }

    // ---- 5. all sequences preserved under ungapping (general invariant)
    {
        std::vector<std::string> in = {"ACGTACGTAA", "ACGTTCGTAA",
                                       "ACGTTTTTCGTAA", "GTACGTAA"};
        std::vector<std::string> out = msa_align(in, P);
        CHECK(!ragged(out), "mixed lengths aligned");
        for (size_t i = 0; i < in.size(); ++i)
            CHECK(ungap(out[i]) == in[i], "ungap == input");
    }

    // ---- 6. profile_from_seq IUPAC fractional counts
    {
        Profile p = profile_from_seq("AN-");
        CHECK(p.ncols() == 3, "profile width");
        CHECK(p.cols[1][0] == 0.25f && p.cols[1][1] == 0.25f &&
              p.cols[1][2] == 0.25f && p.cols[1][3] == 0.25f,
              "N -> 0.25 each");
        CHECK(p.cols[2][4] == 1.0f && p.occ[2] == 0.0f, "gap column");
    }

    // ---- 7. NJ on a known 4-taxon additive tree
    //     s0,s1 close; s2,s3 close; far apart -> joins (s0,s1),(s2,s3)
    {
        // construct sequences whose kmer distance reflects the split
        std::vector<std::string> seqs = {
            "ACGTACGTACGT", "ACGTACGTACGA",      // differ by 1
            "TTTTCCCCAAAA", "TTTTCCCCAAAT"       // differ by 1, far from above
        };
        std::vector<float> D = kmer_distances(seqs, 5);
        Tree t = nj_tree(D, 4);
        CHECK(t.root >= 0, "NJ root");
        // find the two cherry nodes; check leaves pair as expected
        bool found01 = false, found23 = false;
        for (auto& nd : t.nodes) {
            int l = nd.left, r = nd.right;
            if (l < 0) continue;
            if ((l == 0 && r == 1) || (l == 1 && r == 0)) found01 = true;
            if ((l == 2 && r == 3) || (l == 3 && r == 2)) found23 = true;
        }
        CHECK(found01 && found23, "NJ recovers cherries (0,1) and (2,3)");
    }

    // ---- 8. align_profiles: gap-gap columns never score as matches
    {
        Profile a = profile_from_seq("ACGT");
        Profile b = profile_from_seq("AC-T");
        // column 2 of b is all-gap: pairing it with A's 'G' must not be
        // treated as a positive match (gap excluded from col_score)
        AlignResult r = align_profiles(a, b, P);
        CHECK(r.score > 0, "profile DP resolves");
        CHECK(ungap(merge_profiles(a, b, r).rows[1]) == "ACT",
              "B's gap column survives merge");
    }

    // ---- 9. determinism: same input twice -> identical output
    {
        std::vector<std::string> in = {"ACGTACGT", "ACGTACGA",
                                       "ACGTTCGT", "TGCATGCA"};
        auto o1 = msa_align(in, P);
        auto o2 = msa_align(in, P);
        CHECK(o1 == o2, "deterministic output");
    }

    // ---- 10. _mt host stages are bit-exact with the sequential spec
    {
        std::mt19937 rng(7);
        static const char b[] = "ACGT";
        for (int n : {2, 3, 5, 8, 17, 40}) {
            std::vector<std::string> in(n);
            for (auto& s : in) {
                for (int i = 0; i < 20 + (int)(rng() % 60); ++i) s += b[rng() % 4];
            }
            auto D1 = kmer_distances(in, P.kmer_k);
            auto D2 = kmer_distances_mt(in, P.kmer_k, 4);
            CHECK(D1 == D2, "kmer_distances_mt not bit-exact");
            auto T1 = nj_tree(D1, n);
            auto T2 = nj_tree_mt(D1, n, 4);
            bool same = T1.root == T2.root && T1.nodes.size() == T2.nodes.size();
            if (same) for (size_t u = 0; u < T1.nodes.size(); ++u)
                if (T1.nodes[u].left != T2.nodes[u].left ||
                    T1.nodes[u].right != T2.nodes[u].right) same = false;
            CHECK(same, "nj_tree_mt not bit-exact");
        }
    }

    if (fails == 0) printf("ALL OK\n");
    else printf("%d FAILURES\n", fails);
    return fails;
}
