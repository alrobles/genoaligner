// genoaligner — SW CIGAR validation (Fase B).
//
// A Smith-Waterman CIGAR is NOT like the WFA one in what it must consume: it covers
// only the ALIGNED substrings, text[start_i..end_i] and pattern[start_j..end_j],
// not the whole sequences. So the checks here take the reported span explicitly.
//
// The three properties under test (same discipline as tests/parity/cigar.hpp):
//   (b) well-formed  the ops consume exactly the aligned span; every 'M' sits on
//                    equal characters and every 'X' on differing ones -- counts
//                    alone proved insufficient on the WFA side (an 'M' over a
//                    mismatch rescores fine and is still wrong).
//   (a) re-score     score(CIGAR) under the SW affine scheme == kernel score.
//                    A run of k consecutive I's or D's is ONE gap of length k and
//                    costs gap_open + (k-1)*gap_extend -- the recurrence's own
//                    convention (E[i][j] = H[i][j-1] - go opens at length 1).
//   (c) external     SeqAn3's local score must equal ours. Lives in the GPU job
//                    (tests/sw/seqan3_sw_oracle.cpp), because SeqAn3 is a C++23
//                    header library only the cluster toolchain builds.
//
// Ops are the shared codes: 0=M 1=X 2=I (consumes text) 3=D (consumes pattern).
// The kernel emits them REVERSED (element 0 = last op); sw_cigar_from_rev gives
// the read-direction string.

#ifndef GENOALIGNER_SW_CIGAR_HPP
#define GENOALIGNER_SW_CIGAR_HPP

#include <string>

namespace genoaligner {

inline std::string sw_cigar_from_rev(const int* rev, int used)
{
    std::string out;
    if (used <= 0) return out;
    out.resize((size_t)used);
    for (int k = 0; k < used; ++k)
        out[(size_t)k] = wfa_op_to_char(rev[used - 1 - k]);   // same 4-code table
    return out;
}

struct SWCigarCheck {
    bool wellformed  = false;
    bool rescore_ok  = false;
    int  rescored    = 0;
    int  consumed_i  = 0;    // text chars consumed (should equal end_i-start_i+1)
    int  consumed_j  = 0;    // pattern chars
    char first_bad_op = 0;
    int  first_bad_i = -1;   // text index of the first violating op
    int  first_bad_j = -1;   // pattern index
};

// Walk a forward CIGAR string over the aligned span of (text, pattern).
// start_i/start_j are 0-based first aligned chars; the CIGAR must land exactly
// on (end_i+1, end_j+1).
inline SWCigarCheck sw_cigar_check(const std::string& cigar,
                                   const std::string& text, const std::string& pattern,
                                   int start_i, int start_j, int end_i, int end_j,
                                   int match, int mismatch, int gap_open, int gap_extend,
                                   int kernel_score)
{
    SWCigarCheck r;
    int i = start_i, j = start_j;
    int score = 0;
    int gap_run = 0;         // +k inside an I run, -k inside a D run, 0 otherwise

    auto close_gap = [&] {
        if (gap_run != 0) {
            const int len = (gap_run > 0) ? gap_run : -gap_run;
            score -= gap_open + (len - 1) * gap_extend;   // penalties are positive, subtracted
            gap_run = 0;
        }
    };

    for (char op : cigar) {
        switch (op) {
            case 'M':
                close_gap();
                if (i > end_i || j > end_j ||
                    i >= (int)text.size() || j >= (int)pattern.size() ||
                    text[(size_t)i] != pattern[(size_t)j]) {
                    if (!r.first_bad_op) { r.first_bad_op = op; r.first_bad_i = i; r.first_bad_j = j; }
                }
                ++i; ++j; score += match;
                break;
            case 'X':
                close_gap();
                if (i > end_i || j > end_j ||
                    i >= (int)text.size() || j >= (int)pattern.size() ||
                    text[(size_t)i] == pattern[(size_t)j]) {
                    if (!r.first_bad_op) { r.first_bad_op = op; r.first_bad_i = i; r.first_bad_j = j; }
                }
                ++i; ++j; score += mismatch;
                break;
            case 'I':
                if (gap_run < 0) close_gap();
                if (i > end_i || i >= (int)text.size()) {
                    if (!r.first_bad_op) { r.first_bad_op = op; r.first_bad_i = i; r.first_bad_j = j; }
                }
                ++i; ++gap_run;
                break;
            case 'D':
                if (gap_run > 0) close_gap();
                if (j > end_j || j >= (int)pattern.size()) {
                    if (!r.first_bad_op) { r.first_bad_op = op; r.first_bad_i = i; r.first_bad_j = j; }
                }
                ++j; --gap_run;
                break;
            default:
                if (!r.first_bad_op) { r.first_bad_op = op ? op : '?'; r.first_bad_i = i; r.first_bad_j = j; }
                break;
        }
    }
    close_gap();

    r.consumed_i  = i - start_i;
    r.consumed_j  = j - start_j;
    r.rescored    = score;
    r.wellformed  = (r.first_bad_op == 0) && (i == end_i + 1) && (j == end_j + 1);
    r.rescore_ok  = r.wellformed && (r.rescored == kernel_score);
    return r;
}

}  // namespace genoaligner

#endif  // GENOALIGNER_SW_CIGAR_HPP
