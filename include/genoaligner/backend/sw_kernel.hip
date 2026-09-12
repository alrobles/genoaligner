// genoaligner — Smith-Waterman (local alignment) kernel.
//
// WHY A SEPARATE FILE, AND WHY THIS PARALLELISATION
// -------------------------------------------------
// WFA (wfa_kernel.hip) computes edit DISTANCE with unit costs and is efficient because
// the wavefront stays narrow: O(n*s) cells, not O(n*m). Smith-Waterman has no such
// structure -- it is a full DP over the matrix with a general scoring scheme -- so the
// classical implementation is the one to write.
//
//   H[i][j] = max(0, H[i-1][j-1] + s(a_i,b_j), E[i][j], F[i][j])
//   E[i][j] = max(H[i][j-1] - gap_open, E[i][j-1] - gap_extend)
//   F[i][j] = max(H[i-1][j] - gap_open, F[i-1][j] - gap_extend)
//
// The max(0, ...) is what makes it LOCAL: a cell that would go negative restarts, so
// the alignment begins and ends anywhere.
//
// CHOICE OF PARALLELISATION, AND WHY NOT ANTI-DIAGONAL
// ----------------------------------------------------
// The textbook GPU approach is an anti-diagonal sweep with one __syncthreads per
// diagonal. This project has the measurement to avoid it: the WFA kernel's per-wavefront
// barrier pattern produced a time that varied 2.5x between launches doing IDENTICAL work
// (docs/RESULTADO_B6_TCUPS.md), because idle threads waiting on barriers get scheduled
// non-deterministically. An anti-diagonal SW kernel has one barrier per anti-diagonal --
// m+n of them per pair -- which is that failure mode, maximised.
//
// Instead: ONE THREAD BLOCK PER PAIR, the block processes the matrix ROW BY ROW, with
// the previous row kept in shared memory. Every thread owns one or more columns and
// walks i = 1..m. Inside a row the cells are independent given the previous row, so
// only ONE barrier per row is needed, and the block size is FIXED (SW_BLOCK) rather
// than derived from the data -- so occupancy cannot vary between launches.
//
// That is m barriers instead of m+n, with a fixed block. It is not the fastest possible
// SW (that needs striped/swizzled layouts), and it is not claimed to be: it is the
// version whose correctness and timing are both checkable.
//
// WHAT THIS FILE DOES NOT DO YET
// ------------------------------
// Traceback. It returns the SCORE and the END coordinate. Reconstructing the path needs
// the matrix retained and a second pass -- and in this project the traceback is where
// the bugs have lived (the WFA CIGAR path never worked through the API while its test
// passed). It gets its own gate before it is claimed.

#ifndef GENOALIGNER_SW_KERNEL_HIP
#define GENOALIGNER_SW_KERNEL_HIP

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

namespace genoaligner {

// Affine gap penalty. Smith-Waterman needs an explicit scheme: the unit costs WFA uses
// are a special case of the recurrence, and under SW with match=0 the maximum is 0 for
// every input (see tests/sw/test_sw_reference.cpp, where that cost a round).
struct SWParams {
    int match;        // score when a_i == b_j       (must be POSITIVE for SW to mean anything)
    int mismatch;     // score when a_i != b_j       (typically negative)
    int gap_open;     // penalty to OPEN a gap       (positive; subtracted)
    int gap_extend;   // penalty to EXTEND a gap     (positive; subtracted)
};

// One pair. Pointers are DEVICE pointers.
struct SWPairView {
    const char* text;        // rows:    i in [0, text_len)
    int         text_len;
    const char* pattern;     // columns: j in [0, pattern_len)
    int         pattern_len;
};

// The best local alignment's score and where it ENDS. The score says whether there is
// a hit; the end coordinate says where.
struct SWResult {
    int score;
    int end_i;   // text index of the last aligned cell (-1 if score == 0)
    int end_j;   // pattern index
};

// Fixed block size. NOT derived from the data: a block sized from the input is how the
// WFA kernel ended up with variable idle occupancy and non-deterministic timing.
static constexpr int SW_BLOCK = 128;

// Score only (no traceback), one block per pair.
//
// Shared memory the caller must provide, in bytes:
//     (pattern_len + 3) * 3 * sizeof(int)
// for the previous H row and the E/F gap arrays. Sizing belongs in ONE place (the Fase 6
// lesson: a launch that re-derives buffer sizes passes the wrong one).
__global__ void sw_score_kernel(const SWPairView* __restrict__ pairs,
                                SWParams           params,
                                SWResult* __restrict__ out);

}  // namespace genoaligner

#endif  // GENOALIGNER_SW_KERNEL_HIP
