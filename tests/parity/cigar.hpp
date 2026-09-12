// genoaligner — Fase 4: shared CIGAR encode/decode, host AND device.
//
// Why this file exists: the walk produces a CIGAR as a reversed array of small
// integer codes. Both the on-device kernel and the CPU-shim harness must
// interpret those codes the same way, and the validation helpers must apply the
// same three checks on both paths. If either side had its own copy, a bug in the
// copy would read as a green GPU run.
//
// Codes: 0=M 1=X 2=I 3=D. The walk appends in REVERSE (last operation first), so
// element 0 of the array is the LAST operation of the alignment.
//
// The three checks (see also tests/parity/cigar.hpp):
//   (a) re-score      score(CIGAR) == kernel score
//   (b) well-formed   the CIGAR consumes exactly len(pattern) and len(text), and
//                     every M sits on a matching pair, every X on a differing one.
//                     Counts alone are NOT enough: three development versions of
//                     the walk consumed the right counts AND re-scored correctly
//                     while emitting 'M' over non-matching positions.
//   (c) external      edlib's own CIGAR must score the same. Needs Python, so it
//                     lives in tests/parity/check_cigar.py and runs from the gate.
//
// Ties are real: distinct CIGARs can both be optimal, and edlib may pick a
// different one. (a) and (b) therefore check OUR CIGAR; (c) compares SCORES,
// never operation strings.

#ifndef GENOALIGNER_CIGAR_HPP
#define GENOALIGNER_CIGAR_HPP

#include <string>

namespace genoaligner {

// Operation codes.
//   0 = M (match)      1 = X (mismatch)    2 = I (insertion)   3 = D (deletion)
constexpr char WFA_OP_CHAR[4] = { 'M', 'X', 'I', 'D' };

inline char wfa_op_to_char(int op) {
    return (op >= 0 && op <= 3) ? WFA_OP_CHAR[op] : '?';
}

// Capacity bound for the reversed code array. At smax=64 the longest possible
// path is 2*smax+1 = 129 operations, so 16384 is very generous; the point of a
// named bound is that the cap must GATE the emit rather than silently corrupt.
// Defined in the kernel header (which both harnesses include) and reused here.
#ifndef WFA_CIGAR_MAX
#define WFA_CIGAR_MAX 16384
#endif

// Build the read-direction string from the reversed code array.
inline std::string cigar_from_rev(const int* rev, int used)
{
    std::string out;
    if (used <= 0) return out;
    out.resize((size_t)used);
    for (int j = 0; j < used; ++j) out[(size_t)j] = wfa_op_to_char(rev[used - 1 - j]);
    return out;
}

struct CigarCheck {
    bool wellformed  = false;   // (b)
    bool rescore_ok  = false;   // (a), requires (b)
    int  rescored    = 0;
    int  first_bad_i = -1;      // pattern index of the first violating op
    int  first_bad_j = -1;      // text index
    char first_bad_op = 0;
};

// (a) + (b): walk the CIGAR over {M,X,I,D}. 'M' must sit on matching characters
// and 'X' on differing ones; the walk must land exactly on (m, n).
inline CigarCheck cigar_check(const std::string& cigar,
                              const std::string& pattern,
                              const std::string& text)
{
    CigarCheck r;
    const int m = (int)pattern.size(), n = (int)text.size();
    int i = 0, j = 0, score = 0;

    for (char op : cigar) {
        switch (op) {
            case 'M':
                if (i >= m || j >= n || pattern[(size_t)i] != text[(size_t)j]) {
                    if (!r.first_bad_op) { r.first_bad_op = op; r.first_bad_i = i; r.first_bad_j = j; }
                }
                ++i; ++j;
                break;
            case 'X':
                if (i >= m || j >= n || pattern[(size_t)i] == text[(size_t)j]) {
                    if (!r.first_bad_op) { r.first_bad_op = op; r.first_bad_i = i; r.first_bad_j = j; }
                }
                ++i; ++j; ++score;
                break;
            case 'I':
                if (j >= n) { if (!r.first_bad_op) { r.first_bad_op = op; r.first_bad_i = i; r.first_bad_j = j; } }
                ++j; ++score;
                break;
            case 'D':
                if (i >= m) { if (!r.first_bad_op) { r.first_bad_op = op; r.first_bad_i = i; r.first_bad_j = j; } }
                ++i; ++score;
                break;
            default:
                if (!r.first_bad_op) { r.first_bad_op = op ? op : '?'; r.first_bad_i = i; r.first_bad_j = j; }
                break;
        }
    }

    r.rescored   = score;
    r.wellformed = (r.first_bad_op == 0) && (i == m) && (j == n);
    r.rescore_ok = r.wellformed && (r.rescored >= 0);
    return r;
}

// Present a CIGAR in run-length form for readability: 3M1X2I.
inline std::string cigar_run_length(const std::string& cigar)
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
    return out.empty() ? std::string("-") : out;
}

} // namespace genoaligner

#endif // GENOALIGNER_CIGAR_HPP
