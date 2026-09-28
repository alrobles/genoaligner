// Iterative refinement (msa_iter_refine, Params::iter_refine) -- the
// MAFFT FFT-NS-i-class tree-bipartition pass. Cases assert the contract:
// valid rows, preserved letters, monotonic objective, determinism.
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

static std::string ungap(std::string s) {
    s.erase(std::remove(s.begin(), s.end(), '-'), s.end());
    return s;
}
static bool ragged(const std::vector<std::string>& msa) {
    if (msa.empty()) return false;
    for (auto& r : msa) if (r.size() != msa[0].size()) return true;
    return false;
}

// Divergent synthetic family: 60% substitution over a random base plus a
// planted conserved block at varying offsets and small indels -- the kind
// of input a greedy progressive order can place suboptimally.
static std::vector<std::string> mk_family(int n, int L, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<std::string> out;
    const char* al = "ACGT";
    for (int i = 0; i < n; ++i) {
        std::string s(L, 'A');
        for (int c = 0; c < L; ++c) if ((g() & 3) < 2) s[c] = al[g() & 3];
        const std::string blk = "GGGCCCAAATTT";
        const int pos = 10 + (int)(g() % 20);
        for (int k = 0; k < 12 && pos + k < L; ++k) s[pos + k] = blk[k];
        if (i % 3 == 0) s.insert(30, "AC");
        if (i % 4 == 0) s.erase(60, 3);
        out.push_back(s);
    }
    return out;
}

int main() {
    Params P;

    // ---- 1. refine improves (never worsens) the SP objective ----------
    {
        auto seqs = mk_family(12, 80, 7);
        Tree t;
        auto a0 = msa_align(seqs, P, &t);
        RefineStats st;
        auto a1 = msa_iter_refine(a0, t, P, 4, &st);
        CHECK(st.obj1 >= st.obj0,
              "refined objective must not drop below the input objective");
        CHECK(st.obj0 == msa_sp_score(a0, P) &&
              st.obj1 == msa_sp_score(a1, P),
              "stats must report the recomputed objectives");
        CHECK(!ragged(a1), "refined rows must stay equal-length");
        bool letters = true;
        for (size_t i = 0; i < seqs.size(); ++i)
            if (ungap(a1[i]) != ungap(seqs[i])) letters = false;
        CHECK(letters, "refinement must preserve each row's letters");
        CHECK(st.rounds >= 1 && st.tried > 0, "stats must count the sweep");
    }

    // ---- 2. deterministic + convergent --------------------------------
    {
        auto seqs = mk_family(10, 70, 11);
        Tree t;
        auto a0 = msa_align(seqs, P, &t);
        auto r1 = msa_iter_refine(a0, t, P, 4);
        auto r2 = msa_iter_refine(a0, t, P, 4);
        CHECK(r1 == r2, "same input must refine to the same output");
        RefineStats st2;
        msa_iter_refine(r1, t, P, 4, &st2);
        CHECK(st2.accepted == 0,
              "a converged MSA accepts no further bipartitions");
    }

    // ---- 3. Params::iter_refine wires through msa_align ---------------
    {
        auto seqs = mk_family(10, 70, 11);
        Tree t;
        Params P2 = P; P2.iter_refine = 4;
        auto via_align = msa_align(seqs, P2);
        auto a0 = msa_align(seqs, P, &t);
        auto manual = msa_iter_refine(a0, t, P, 4);
        CHECK(via_align == manual,
              "P.iter_refine through msa_align must equal the manual pass");
        // and a with_tree call refines identically
        auto via_tree = msa_align_with_tree(seqs, t, P2);
        CHECK(via_tree == manual,
              "P.iter_refine through msa_align_with_tree must match");
    }

    // ---- 4. degenerate cases ------------------------------------------
    {
        // identical seqs: objective already maximal, refine is a no-op
        std::vector<std::string> same(4, "ACGTACGT");
        auto a0 = msa_align(same, P);
        Tree t;
        {
            std::vector<float> D =
                kmer_distances(same, P.kmer_k, P.alpha);
            t = nj_tree(D, (int)same.size());
        }
        RefineStats st;
        auto r = msa_iter_refine(a0, t, P, 2, &st);
        CHECK(r == a0 && st.accepted == 0,
              "a perfect MSA must be a fixed point");
        // 1 and 2 seqs: refine must never run / never crash
        auto one = msa_iter_refine({"ACGT"}, Tree{}, P, 4);
        CHECK(one.size() == 1 && one[0] == "ACGT",
              "single row passes through untouched");
    }

    // ---- 5. sp_score semantics ----------------------------------------
    {
        Params q = P;
        CHECK(msa_sp_score({"ACGT"}, q) == 0.0, "one row scores 0");
        CHECK(msa_sp_score({"ACGT", "ACGT"}, q) > 0.0,
              "identical rows score positive");
        CHECK(msa_sp_score({"ACGT", "A-GT"}, q) <
              msa_sp_score({"ACGT", "ACGT"}, q),
              "a gap must lower the objective");
        // two separate 1-gaps cost more than one 2-gap (affine)
        CHECK(msa_sp_score({"ACGTT", "A-G-T"}, q) <
              msa_sp_score({"ACGTT", "A--TT"}, q),
              "two gap runs must cost more than one extended run");
    }

    // ---- 6. codon-token rows refine ------------------------------------
    {
        // alpha=65 tokens (128+idx); refine must treat them like any
        // alphabet. Build a tiny codon family via codon_encode.
        Params C = codon_params(1);
        std::vector<std::string> nt = {
            "ATGAAACCCGGGTTTAAACCCGGGTTT",
            "ATGAAACCGGGGTTTAAACCCGGGTTT",
            "ATGAAACCCGGGTTGAAACCCGGGTTT",
            "ATGAAACCCGGGTTTAAACCCGAGTTT",
            "ATGTAACCCGGGTTTAAACCCGGGTTT" };
        auto tok = codon_encode(nt, 1);
        Tree t;
        auto a0 = msa_align(tok, C, &t);
        auto a1 = msa_iter_refine(a0, t, C, 2);
        CHECK(!ragged(a1), "codon rows must stay equal-length");
        CHECK(msa_sp_score(a1, C) >= msa_sp_score(a0, C),
              "codon refine must not regress the objective");
    }

    if (fails == 0) { printf("iter-refine tests: PASS\n"); return 0; }
    printf("iter-refine tests: %d FAIL\n", fails);
    return 1;
}
