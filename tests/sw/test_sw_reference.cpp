// genoaligner — Smith-Waterman: reference implementation + expectations.
//
// ORDER OF WORK, AND WHY
// ----------------------
// This project has learned that a GPU kernel is easy to get subtly wrong and that the
// cheap gate comes FIRST. So this file provides a straightforward O(mn) Smith-Waterman
// in plain host C++ -- written to be obviously correct rather than fast -- and pins it
// against hand-computed expectations on the cases that break local aligners. The kernel
// is compared against THIS reference next.
//
// The reference is deliberately a different code path from the kernel: a full matrix
// with explicit loops, versus the kernel's two buffered antidiagonals. Same formula,
// different layout -- which is what makes agreement meaningful.
//
// A CORRECTION THAT COST A ROUND, RECORDED BECAUSE IT IS INSTRUCTIVE
// -----------------------------------------------------------------
// The first version of this file used the edit-distance-like scheme {match=0,
// mismatch=-1} -- the same one WFA uses -- and expected positive scores. It got 0
// everywhere, and that is CORRECT: with match scoring 0, no alignment can score above
// zero, so the maximum is 0 for every input. Smith-Waterman with match=0 is degenerate.
// SW needs a POSITIVE match reward; the unit-cost scheme is a special case of the
// dynamic program, not a usable instance of it.
//
// The same round also had a wrong expectation for the positive-match case: a core
// flanked by identical bases scores the WHOLE identical region, not just the core,
// because the flanks match too. Both were errors in the expectations, not in the code
// -- which is the whole reason to compute expectations by hand before trusting a green
// test.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace sw_ref {

struct Params { int match, mismatch, gap_open, gap_extend; };

struct Outcome {
    int score = 0;
    int end_i = -1;   // text index of the last aligned cell
    int end_j = -1;   // pattern index
};

// Straightforward full-matrix Smith-Waterman. Correct beats fast here.
inline Outcome align(const std::string& text, const std::string& pattern, Params p)
{
    const int m = (int)text.size();
    const int n = (int)pattern.size();
    Outcome r;
    if (m == 0 || n == 0) return r;

    std::vector<int> H((size_t)(m + 1) * (size_t)(n + 1), 0);
    std::vector<int> E((size_t)(m + 1) * (size_t)(n + 1), 0);
    std::vector<int> F((size_t)(m + 1) * (size_t)(n + 1), 0);
    auto at = [&](std::vector<int>& M, int i, int j) -> int& {
        return M[(size_t)i * (size_t)(n + 1) + (size_t)j];
    };

    int best = 0, bi = -1, bj = -1;
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            const int s = (text[(size_t)i - 1] == pattern[(size_t)j - 1]) ? p.match : p.mismatch;
            int e = at(H, i, j - 1) - p.gap_open;
            const int e2 = at(E, i, j - 1) - p.gap_extend;
            if (e2 > e) e = e2;
            int f = at(H, i - 1, j) - p.gap_open;
            const int f2 = at(F, i - 1, j) - p.gap_extend;
            if (f2 > f) f = f2;

            int h = at(H, i - 1, j - 1) + s;
            if (e > h) h = e;
            if (f > h) h = f;
            if (h < 0) h = 0;                 // LOCAL: restart rather than go negative
            at(H, i, j) = h;
            at(E, i, j) = e;
            at(F, i, j) = f;

            if (h > best) { best = h; bi = i - 1; bj = j - 1; }
        }
    }
    r.score = best;
    r.end_i = bi;
    r.end_j = bj;
    return r;
}

// A GLOBAL alignment, for the comparison that gives LOCAL its meaning: the same input
// under NW must score lower (or worse) because it is forced to pay for the flanks.
inline int global_score(const std::string& text, const std::string& pattern, Params p)
{
    const int m = (int)text.size(), n = (int)pattern.size();
    if (m == 0 || n == 0) return 0;
    const int NEG = -1000000;
    std::vector<int> H((size_t)(m + 1) * (size_t)(n + 1));
    std::vector<int> E((size_t)(m + 1) * (size_t)(n + 1));
    std::vector<int> F((size_t)(m + 1) * (size_t)(n + 1));
    auto at = [&](std::vector<int>& M, int i, int j) -> int& {
        return M[(size_t)i * (size_t)(n + 1) + (size_t)j];
    };
    for (int i = 0; i <= m; ++i) { at(H,i,0) = -p.gap_open - (i-1 > 0 ? (i-1)*p.gap_extend : 0); at(E,i,0)=NEG; at(F,i,0)=NEG; }
    for (int j = 0; j <= n; ++j) { at(H,0,j) = -p.gap_open - (j-1 > 0 ? (j-1)*p.gap_extend : 0); at(E,0,j)=NEG; at(F,0,j)=NEG; }
    at(H,0,0) = 0;
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            const int s = (text[(size_t)i-1] == pattern[(size_t)j-1]) ? p.match : p.mismatch;
            int e = at(H,i,j-1) - p.gap_open;
            const int e2 = at(E,i,j-1) - p.gap_extend; if (e2 > e) e = e2;
            int f = at(H,i-1,j) - p.gap_open;
            const int f2 = at(F,i-1,j) - p.gap_extend; if (f2 > f) f = f2;
            int h = at(H,i-1,j-1) + s;
            if (e > h) h = e;
            if (f > h) h = f;
            at(H,i,j)=h; at(E,i,j)=e; at(F,i,j)=f;
        }
    }
    return at(H,m,n);
}

}  // namespace sw_ref

static int g_fail = 0;

static void expect(int got, int want, const char* what)
{
    if (got != want) { printf("      FAIL: %s -- got %d, want %d\n", what, got, want); ++g_fail; }
}

static void row(const std::string& text, const std::string& pattern,
                const sw_ref::Params& p, const char* label, int want)
{
    const sw_ref::Outcome r = sw_ref::align(text, pattern, p);
    printf("  %-44s score=%4d end=(%d,%d)\n", label, r.score, r.end_i, r.end_j);
    expect(r.score, want, label);
}

int main()
{
    printf("genoaligner — Smith-Waterman reference (local) + expectations\n");

    // SW needs a POSITIVE match. This scheme is the working instance; the unit-cost
    // scheme WFA uses is degenerate under SW (see the header).
    const sw_ref::Params SWP{1, -1, 2, 1};      // match +1, mismatch -1, gap open 2, extend 1
    printf("  params: match=%d mismatch=%d gap_open=%d gap_extend=%d\n\n",
           SWP.match, SWP.mismatch, SWP.gap_open, SWP.gap_extend);

    // ---- 1. THE defining property: local does not pay for the flanks ------
    printf("-- shared core, divergent flanks (this is what makes it LOCAL) --\n");
    {
        const std::string core = "ACGTACGTACGT";   // 12 matches -> 12 * 1 = 12
        const std::string t = std::string(10, 'T') + core + std::string(10, 'G');
        const std::string q = std::string(10, 'C') + core + std::string(10, 'A');
        row(t, q, SWP, "core only -> 12", 12);

        const int g = sw_ref::global_score(t, q, SWP);
        printf("      global (NW) on the SAME pair: %d\n", g);
        if (!(g < 12)) {
            printf("      FAIL: local must beat global here, or the test proves nothing\n");
            ++g_fail;
        }
    }

    // ---- 2. identical inputs: everything matches -------------------------
    printf("\n-- identical and trivial inputs --\n");
    {
        row("ACGT", "ACGT", SWP, "4 identical -> 4", 4);
        row("A", "A", SWP, "single base -> 1", 1);
        row(std::string(20, 'A'), std::string(20, 'A'), SWP, "20 identical -> 20", 20);
    }

    // ---- 3. no shared base: 0, never negative ----------------------------
    printf("\n-- no shared subsequence --\n");
    {
        row("AAAAAAAA", "CCCCCCCC", SWP, "disjoint alphabets -> 0", 0);
        // NOTE the first version of this case used "ACGTACGT" vs "TGCATGCA" and
        // expected 0. That is WRONG: they share 'A' and 'C', so a single match scores
        // 1 and the reference was right to report 1. Checked by hand: the pattern has
        // an A at index 3, the text has an A at index 0 -- one match, score 1.
        row("ACGTACGT", "TGCATGCA", SWP, "shares one 'A' -> 1 (not 0)", 1);
        // A genuinely disjoint pair over a 2-letter alphabet: no base can match.
        row("AAAAAA", "CCCCCC", SWP, "purely disjoint -> 0", 0);
    }

    // ---- 4. a mismatch costs, but a lone mismatch cannot beat 0 ----------
    printf("\n-- mismatch alone --\n");
    {
        row("A", "C", SWP, "single mismatch -> 0 (not -1)", 0);
    }

    // ---- 5. boundaries --------------------------------------------------
    printf("\n-- the core at each boundary --\n");
    {
        const std::string core = "ACGTACGT";   // 8 -> 8
        row(core + std::string(20, 'T'), core + std::string(20, 'G'), SWP, "core at start -> 8", 8);
        row(std::string(20, 'T') + core, std::string(20, 'G') + core, SWP, "core at end -> 8", 8);
    }

    // ---- 6. a gap is worthwhile when it buys more matches ----------------
    printf("\n-- gap handling --\n");
    {
        // text has an insertion of 2 bases inside an otherwise matching 8: matching
        // through it needs one gap of length 2 = gap_open + gap_extend = 3, and buys
        // 8 matches = 8. Net 8 - 3 = 5 under local, but local may also just take the
        // longer side. 8 matches > (4 matches - 3), so the gap is taken: 8 - 3 = 5.
        row("ACGTTTACGT", "ACGTACGT", SWP, "one gap of 2 inside a match -> 8-3=5", 5);
    }

    // ---- 7. repeats: the maximum is attained in more than one place ------
    printf("\n-- repeats (multiple optima; score is well-defined, position may tie) --\n");
    {
        row("ACACACAC", "ACAC", SWP, "tandem repeat -> 4", 4);
    }

    // ---- 8. empty input -------------------------------------------------
    printf("\n-- empty input --\n");
    {
        row("", "ACGT", SWP, "empty text -> 0", 0);
        row("ACGT", "", SWP, "empty pattern -> 0", 0);
        row("", "", SWP, "both empty -> 0", 0);
    }

    printf("\n");
    if (g_fail == 0) {
        printf("RESULT: PASS -- SW reference pinned; expectations hand-derived,\n"
               "                and the local-vs-global distinction asserted.\n");
        return 0;
    }
    printf("RESULT: FAIL (%d)\n", g_fail);
    return 1;
}
