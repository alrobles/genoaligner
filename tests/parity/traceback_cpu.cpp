// genoaligner — Fase 4: traceback (CIGAR reconstruction), CPU prototype.
//
// THE PROBLEM. The score kernel answers "how different are these sequences".
// Traceback answers "how, specifically" -- which positions align to which. It
// needs the full wavefront history, not just the final one, because the path is
// recovered by walking BACKWARDS from the end, asking at each step which
// predecessor could have produced the current offset.
//
// ============================================================ THE ALGORITHM
//
// Ported from github.com/smarco/WFA, wavefront/wavefront_backtrace.c
// :: wavefront_backtrace_linear (lines 223-319), specialised to unit costs
// (edit distance). Penalties per wavefront_penalties.c:51:
//
//     match = 0   mismatch = 1   gap_opening1 = 1   gap_extension1 = -1
//
// For the EDIT metric there is only the M wavefront -- the I1/D1 component
// wavefronts of the affine path do not exist. So all three predecessors are read
// from M[score-1]:
//
//     misms = M[s-1][k]   + 1     substitution, consumes one of each
//     ins   = M[s-1][?]   + 1     insertion, consumes a text char
//     del   = M[s-1][?]   + 1     deletion,  consumes a pattern char
//
// ------------------------------------------------------------------ TRAP (1)
// The k+/-1 direction for ins vs del is the exact inversion that cost this
// project a full debugging cycle at R1. The reference's *naming* is not a
// reliable guide; what the operation EMITS is. From backtrace.c:
//
//     case backtrace_I1_open: emit 'I'; --k; --offset;      <- insertion
//     case backtrace_D1_open: emit 'D'; ++k;                <- deletion
//
// and the corresponding source lookups are ins1_open -> offsets[k+1],
// del1_open -> offsets[k-1]. Both are verified here against a DP, not assumed.
//
// ------------------------------------------------------------------ TRAP (2)
// Ties are real and legal. Multiple distinct CIGARs score identically; edlib may
// pick a different one. The tests below therefore compare SCORES, never
// operation strings, and a differing string is not a failure.
//
// ------------------------------------------------------------------ BUILD
//   g++ -O2 -std=c++17 -DGENOALIGNER_HIP_SHIM \
//       -Itests/parity/hip_cpu_shim -I. -o traceback_cpu tests/parity/traceback_cpu.cpp
//
// ============================================================ THE FORWARD PASS
// The wavefronts are produced by wfa_step() from the SHIPPED kernel header, so
// the path is reconstructed from the same arithmetic that already scores 100%
// against three external oracles. A re-typed forward pass could drift from the
// kernel and invalidate the whole test.
#define GENOALIGNER_HIP_SHIM
#include "hip/hip_runtime.h"
#include "include/genoaligner/backend/wfa_kernel.hip"
#include "src/reference/edit_distance_cpu.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace genoaligner;

// Shim launch-configuration globals (the kernel references them; we never
// launch it here, but the symbols must resolve).
uint3 threadIdx{0,0,0};
uint3 blockIdx{0,0,0};
dim3  blockDim{1,1,1};
dim3  gridDim{1,1,1};

// ---------------------------------------------------------------------------
// Forward pass retaining EVERY wavefront, indexed by score.
//
// Memory: (smax+1) wavefronts of (2*smax+3) ints. At smax=64 that is 65*131*4
// = ~34 KB per pair. This is the CPU prototype of the GPU memory strategy: at
// 1000 pairs that is 34 MB, well inside the MI210's 63 GB, so materialising the
// full stack is affordable and far simpler than the reference's piggyback
// scheme (which exists for long sequences, not for this regime).
// ---------------------------------------------------------------------------
struct WavefrontStack {
    int smax;
    int stride;                       // slots per wavefront (2*smax+3)
    std::vector<std::vector<int>> wf; // wf[s][idx(k)], idx(k) = k + smax + 1
    int final_score = -1;

    int idx(int k) const { return k + smax + 1; }
    // M[s][k], or WFA_NEG when out of range at that score.
    int at(int s, int k) const {
        if (s < 0 || s >= (int)wf.size()) return WFA_NEG;
        const int lo = -s, hi = s;
        if (k < lo || k > hi) return WFA_NEG;
        return wf[s][idx(k)];
    }
};

static WavefrontStack forward_all(const std::string& pattern, const std::string& text,
                                  int smax)
{
    const int m = (int)pattern.size(), n = (int)text.size();
    WavefrontStack S;
    S.smax = smax;
    S.stride = 2 * smax + 3;
    S.wf.assign(smax + 1, std::vector<int>(S.stride, WFA_NEG));

    if (m == 0 || n == 0) { S.final_score = (m == 0) ? n : m; return S; }

    // ---- score 0: diagonal k = 0 only, greedy extend ----------------------
    {
        int offset = 0;
        while (offset < m && offset < n && pattern[offset] == text[offset]) ++offset;
        S.wf[0][S.idx(0)] = offset;
        const int alignment_k = wfa_diag(n, m);
        if (alignment_k == 0 && offset >= n) { S.final_score = 0; return S; }
    }

    for (int s = 1; s <= smax; ++s) {
        for (int k = -s; k <= s; ++k) {
            // Same call shape as the shipped kernel: the three predecessors come
            // from the previous wavefront at k-1, k, k+1.
            S.wf[s][S.idx(k)] = wfa_step(pattern.data(), text.data(), k, m, n,
                                         S.wf[s-1][S.idx(k - 1)],   // ins
                                         S.wf[s-1][S.idx(k + 1)],   // del
                                         S.wf[s-1][S.idx(k)]);      // mism
        }
        const int alignment_k = wfa_diag(n, m);
        if (alignment_k >= -s && alignment_k <= s) {
            const int off = S.wf[s][S.idx(alignment_k)];
            if (wfa_reachable(off) && off >= n) { S.final_score = s; return S; }
        }
    }
    return S;   // final_score stays -1: unresolved within smax
}

// ---------------------------------------------------------------------------
// Backward walk. Transcribes wavefront_backtrace_linear, edit-distance case.
// Returns the CIGAR over {M, X, I, D}, or "" if the trace fails.
//
// Coordinate reminder: the stored value is the TEXT coordinate h, and
// v = h - k. 'M' = match, 'X' = mismatch, 'I' = insertion (text char only),
// 'D' = deletion (pattern char only).
// ---------------------------------------------------------------------------
static std::string traceback(const std::string& pattern, const std::string& text,
                             const WavefrontStack& S)
{
    const int m = (int)pattern.size(), n = (int)text.size();
    const int score_total = S.final_score;
    if (score_total < 0) return "";

    if (m == 0 || n == 0) {
        // Degenerate: all insertions or all deletions.
        std::string c;
        for (int i = 0; i < n; ++i) c.push_back('I');
        for (int i = 0; i < m; ++i) c.push_back('D');
        return c;
    }

    const int alignment_k = wfa_diag(n, m);
    const int alignment_offset = S.at(score_total, alignment_k);

    std::string rev;                       // built backwards, reversed at the end
    int score = score_total;
    int k = alignment_k;
    int offset = alignment_offset;

    // End position. For a global alignment ending exactly at (n, m) these are
    // equal and the reference's tail handling is a no-op, but keep it because
    // the degenerate paths reach here.
    int h = wfa_h(k, offset);
    int v = wfa_v(k, offset);

    // Trailing insertions/deletions outside the grid -- reference lines 246-254.
    if (v < m) { for (int i = m - v; i > 0; --i) rev.push_back('D'); }
    if (h < n) { for (int i = n - h; i > 0; --i) rev.push_back('I'); }

    // ---- main walk: reference lines 256-304 -------------------------------
    //
    // THE EXACT SOURCE SHAPES. Read from backtrace.c rather than inferred --
    // this cost three wrong attempts, each of which re-scored correctly while
    // emitting 'M' over positions that do not match.
    //
    //     misms : offsets[k]   + 1     (backtrace.c:73)
    //     del   : offsets[k+1]         (backtrace.c:114)   <-- NO +1
    //     ins   : offsets[k-1] + 1     (backtrace.c:173)   <-- HAS +1
    //
    // The +1 on misms and ins (but NOT del) is what makes
    // `num_matches = offset - max_offset` land correctly:
    //
    //   - misms consumes one position on the SAME diagonal, so the stored
    //     predecessor is one further along; +1 brings it onto the comparison
    //     scale, and the later `--offset` reconciles it.
    //   - ins  does the same: `--k --offset` after picking it.
    //   - del  moves diagonals instead of consuming an offset, so its value is
    //     already directly comparable and needs no +1; `++k` alone reconciles.
    //
    // Getting this wrong does NOT show up as a wrong score or a wrong character
    // count: it shows up as an 'M' sitting on a non-matching pair, which is why
    // the well-formedness check verifies characters and not just counts.
    while (v > 0 && h > 0 && score > 0) {
        const int s_prev = score - 1;      // edit metric: every op costs 1

        const int misms = S.at(s_prev, k) + 1;     // +1, and k
        const int ins   = S.at(s_prev, k - 1) + 1; // +1, and k-1
        const int del   = S.at(s_prev, k + 1);     // no +1, k+1

        // Compare on the reference's own scale. A sentinel must not win.
        int best = WFA_NEG;
        int op = 0;                       // 0=none 1=mism 2=ins 3=del
        if (wfa_reachable(misms) && misms >= best) { best = misms; op = 1; }
        if (wfa_reachable(ins)   && ins   >  best) { best = ins;   op = 2; }
        if (wfa_reachable(del)   && del   >  best) { best = del;   op = 3; }
        if (op == 0) break;                        // no source

        const int num_matches = offset - best;
        for (int i = 0; i < num_matches; ++i) rev.push_back('M');

        offset = best;
        v = wfa_v(k, offset);
        h = wfa_h(k, offset);

        if (op == 1) {
            rev.push_back('X');
            --offset;                          // mismatch consumes one of each
        } else if (op == 2) {
            rev.push_back('I');
            --k; --offset;                     // insertion consumes a text char
        } else {
            rev.push_back('D');
            ++k;                               // deletion consumes a pattern char
        }

        v = wfa_v(k, offset);
        h = wfa_h(k, offset);
        --score;
    }

    // ---- leading matches, then leading indels: reference lines 305-315 -----
    if (v > 0 && h > 0) {
        const int num_matches = std::min(v, h);
        for (int i = 0; i < num_matches; ++i) rev.push_back('M');
        v -= num_matches;
        h -= num_matches;
    }
    while (v > 0) { rev.push_back('D'); --v; }
    while (h > 0) { rev.push_back('I'); --h; }

    std::string out(rev.rbegin(), rev.rend());
    return out;
}

// ---------------------------------------------------------------------------
// The three checks. See the plan: self-consistency alone is weak, so the gate
// uses re-score AND well-formedness AND an external comparison.
// ---------------------------------------------------------------------------

// (a) Re-score the CIGAR independently of the walk.
// M/X -> 1 each; I -> 1 text; D -> 1 pattern. A mismatch ('X') also requires the
// characters to actually differ, else the CIGAR is lying about the alignment.
static int cigar_score(const std::string& cigar,
                       const std::string& pattern, const std::string& text,
                       bool* ops_consistent /*out*/)
{
    int i = 0, j = 0, score = 0;
    bool consistent = true;
    for (char op : cigar) {
        switch (op) {
            case 'M':
                if (i >= (int)pattern.size() || j >= (int)text.size() ||
                    pattern[i] != text[j]) consistent = false;
                ++i; ++j;
                break;
            case 'X':
                if (i >= (int)pattern.size() || j >= (int)text.size() ||
                    pattern[i] == text[j]) consistent = false;
                ++i; ++j; ++score;
                break;
            case 'I':
                if (j >= (int)text.size()) consistent = false;
                ++j; ++score;
                break;
            case 'D':
                if (i >= (int)pattern.size()) consistent = false;
                ++i; ++score;
                break;
            default:
                consistent = false;
                break;
        }
    }
    if (i != (int)pattern.size() || j != (int)text.size()) consistent = false;
    *ops_consistent = consistent;
    return score;
}

// Present a CIGAR in run-length form for readability: 3M1X2I.
static std::string run_length(const std::string& cigar)
{
    std::string out;
    char buf[32];
    size_t i = 0;
    while (i < cigar.size()) {
        size_t j = i;
        while (j < cigar.size() && cigar[j] == cigar[i]) ++j;
        std::snprintf(buf, sizeof(buf), "%zu%c", j - i, cigar[i]);
        out += buf;
        i = j;
    }
    return out.empty() ? "-" : out;
}

static int g_fail = 0;

struct Chk { bool rescore, wellformed; };

static Chk check_case(const std::string& pattern, const std::string& text, int smax)
{
    Chk r{false, false};
    WavefrontStack S = forward_all(pattern, text, smax);
    if (S.final_score < 0) { return r; }          // unresolved: skipped upstream

    const std::string cigar = traceback(pattern, text, S);
    if (cigar.empty() && !(pattern.empty() && text.empty())) return r;

    bool ops_ok = false;
    const int rescored = cigar_score(cigar, pattern, text, &ops_ok);
    r.rescore = (rescored == S.final_score);
    r.wellformed = ops_ok;
    return r;
}

int main(int argc, char** argv)
{
    const int smax = 64;

    // --emit <path>: write index / pattern / text / CIGAR for the external
    // edlib comparison (check_cigar.py). Column 4 carries the CIGAR so the
    // same reader shape works as for the score harness.
    std::string emit_path;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--emit" && i + 1 < argc) emit_path = argv[i + 1];
    }

    // ---- hand cases with expected scores (not expected CIGAR strings) ------
    struct Case { const char* p; const char* t; int score; const char* note; };
    const std::vector<Case> hand = {
        {"ACGT", "ACGT", 0, "identical"},
        {"ACGT", "ACGA", 1, "one mismatch"},
        {"ACGTA", "ACGT", 1, "one deletion"},
        {"ACGT", "ACGTA", 1, "one insertion"},
        {"GG", "TTT", 3, "R1 reproducer"},
        {"C", "GT", 2, "R1 reproducer"},
        {"CT", "CCC", 2, "R1 reproducer"},
        {"CG", "GAC", 3, "R1 reproducer"},
        {"A", "AA", 1, "R1 reproducer (sentinel)"},
        {"", "", 0, "both empty"},
        {"ACGT", "", 4, "empty text"},
        {"", "ACGT", 4, "empty pattern"},
        {"AAAA", "AAAAA", 1, "homopolymer"},
        {"ACGTACGT", "TTTTTTTT", 6, "no match"},
    };

    std::printf("genoaligner Fase 4 -- CPU traceback prototype\n");
    std::printf("  smax = %d\n\n", smax);

    int bad = 0;
    for (const auto& c : hand) {
        std::string P(c.p), T(c.t);
        WavefrontStack S = forward_all(P, T, smax);
        std::string cigar = traceback(P, T, S);
        bool ops_ok = false;
        int rescored = cigar_score(cigar, P, T, &ops_ok);
        bool s_ok = (S.final_score == c.score);
        bool r_ok = ops_ok && (rescored == S.final_score);
        if (!(s_ok && r_ok)) ++bad;
        std::printf("  %-10s vs %-10s score=%2d(w%2d) %-6s cigar=%-16s %s\n",
                    c.p[0] ? c.p : "(empty)", c.t[0] ? c.t : "(empty)",
                    S.final_score, c.score, s_ok ? "ok" : "BAD",
                    run_length(cigar).c_str(),
                    r_ok ? "rescore-ok" : "** RESCORE FAIL **");
        if (!ops_ok)
            std::printf("      note: CIGAR does not consume exactly the inputs\n");
    }

    // ---- random sweep over the control-set shape --------------------------
    // Report the two checks SEPARATELY: a combined number hides which one broke.
    std::mt19937 rng(7);
    const char A[] = "ACGT";
    std::uniform_int_distribution<int> ch(0, 3);
    int n_tot = 0, n_rescore = 0, n_wf = 0, n_skip = 0;
    int shown = 0;

    FILE* emit = nullptr;
    if (!emit_path.empty()) {
        emit = std::fopen(emit_path.c_str(), "w");
        if (!emit) {
            std::fprintf(stderr, "ERROR: cannot open emit path %s\n", emit_path.c_str());
            return 2;
        }
        std::fprintf(emit, "# genoaligner Fase 4 CIGAR emit\n");
        std::fprintf(emit, "index\tpattern\ttext\tcigar\tscore\n");
    }

    for (int cases = 0; cases < 3000; ++cases) {
        const int len = 24 + (cases % 6) * 8;
        std::string t;
        for (int i = 0; i < len; ++i) t.push_back(A[ch(rng)]);
        std::string p = t;
        std::uniform_int_distribution<int> pos(0, len - 1);
        const int mode = cases % 3;
        if (mode == 0) { for (int i = 0; i < len/4; ++i) p[pos(rng)] = A[ch(rng)]; }
        else if (mode == 1) {
            for (int i = 0; i < 3; ++i) if (p.size() > 6) p.erase(p.begin() + pos(rng) % p.size());
            for (int i = 0; i < 3; ++i) p.insert(p.begin() + pos(rng) % p.size(), A[ch(rng)]);
        } else {
            for (int i = 0; i < len/8; ++i) p[pos(rng)] = A[ch(rng)];
            if (p.size() > 6) {
                p.erase(p.begin() + (pos(rng) % p.size()));
                p.insert(p.begin() + (pos(rng) % p.size()), A[ch(rng)]);
            }
        }

        WavefrontStack S = forward_all(p, t, smax);
        if (S.final_score < 0) { ++n_skip; continue; }
        ++n_tot;

        const std::string cigar = traceback(p, t, S);
        bool ops_ok = false;
        const int rescored = cigar_score(cigar, p, t, &ops_ok);

        if (emit) {
            std::fprintf(emit, "%d\t%s\t%s\t%s\t%d\n",
                         cases, p.c_str(), t.c_str(),
                         cigar.empty() ? "-" : cigar.c_str(), S.final_score);
        }

        if (ops_ok && rescored == S.final_score) ++n_rescore; else {
            if (shown++ < 3) {
                std::printf("  RESCORE FAIL mode=%d m=%zu n=%zu score=%d rescored=%d ops_ok=%d\n",
                            mode, p.size(), t.size(), S.final_score, rescored, (int)ops_ok);
                std::printf("    pattern = %s\n", p.c_str());
                std::printf("    text    = %s\n", t.c_str());
                std::printf("    cigar   = %s\n", run_length(cigar).c_str());
                // Show the first M/X that violates consistency.
                {
                    int i = 0, j = 0; bool found = false;
                    for (char op : cigar) {
                        if (op == 'M' && i < (int)p.size() && j < (int)t.size() && p[i] != t[j]) {
                            std::printf("    first bad M at pattern[%d]=%c text[%d]=%c\n",
                                        i, p[i], j, t[j]);
                            found = true; break;
                        }
                        if (op == 'M' || op == 'X') { ++i; ++j; }
                        else if (op == 'I') ++j;
                        else if (op == 'D') ++i;
                    }
                    if (!found) std::printf("    (all M consistent; failure is in counts)\n");
                }
            }
        }
        if (ops_ok) ++n_wf; else {
            if (shown++ < 5)
                std::printf("  WELLFORMED FAIL mode=%d m=%zu n=%zu cigar=%s\n",
                            mode, p.size(), t.size(), run_length(cigar).c_str());
        }
    }

    std::printf("\nrandom sweep (3000 cases, smax=%d)\n", smax);
    std::printf("  resolved            : %d\n", n_tot);
    std::printf("  skipped (isD > smax): %d\n", n_skip);
    std::printf("  rescore == score    : %d/%d\n", n_rescore, n_tot);
    std::printf("  well-formed CIGAR   : %d/%d\n", n_wf, n_tot);

    const bool pass = (bad == 0) && (n_rescore == n_tot) && (n_wf == n_tot) && (n_tot > 0);
    std::printf("\n==== Fase 4 CPU %s ====\n",
                pass ? "TRACEBACK OK" : "TRACEBACK FAILED");

    if (emit) {
        std::fclose(emit);
        std::printf("emitted CIGARs: %s\n", emit_path.c_str());
    }
    return pass ? 0 : 1;
}
