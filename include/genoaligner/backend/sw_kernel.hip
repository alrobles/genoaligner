// genoaligner — Smith-Waterman (local alignment) kernel.
//
// WHY A SEPARATE FILE, AND WHY ANTIDIAGONAL
// -----------------------------------------
// WFA (wfa_kernel.hip) computes edit DISTANCE with unit costs and is efficient
// precisely because the wavefront stays narrow: it visits O(n*s) cells, not O(n*m).
// Smith-Waterman has no such structure. It is a full DP over the matrix with a
// general scoring scheme, so the honest implementation is the classical one:
//
//   H[i][j] = max(0, H[i-1][j-1] + s(a_i, b_j), E[i][j], F[i][j])
//   E[i][j] = max(H[i][j-1] - gap_open_extend, E[i][j-1] - gap_extend)
//   F[i][j] = max(H[i-1][j] - gap_open_extend, F[i-1][j] - gap_extend)
//
// with the max(0, ...) being what makes it LOCAL: a cell that would score negative
// restarts at zero, so the alignment begins and ends anywhere.
//
// Parallelism: antidiagonal. Every cell (i, j) depends on (i-1, j-1), (i, j-1) and
// (i-1, j) -- all on the PREVIOUS two antidiagonals -- so all cells of one
// antidiagonal are independent. One block per pair walks the antidiagonals in
// sequence; the threads within a block cover the cells of the current one.
//
// PARALLELISATION NOTE, LEARNED THE HARD WAY IN THIS PROJECT
// ---------------------------------------------------------
// blockDim is a FIXED number of threads, NOT 2*len+1. The WFA kernel sized its block
// from the search bound and, as measured in docs/RESULTADO_FASE7_OPTIMIZACION.md,
// that left 76-97% of the threads idle on a barrier and produced a NON-DETERMINISTIC
// time (2.5x between launches doing identical work, verified in
// docs/RESULTADO_B6_TCUPS.md). This kernel uses a fixed block and grid-strides the
// antidiagonal, so idle occupancy cannot vary between launches.
//
// WHAT IS NOT HERE YET
// --------------------
// Traceback. This file computes the SCORE and the end coordinate (where the maximum
// was found), which is what a caller needs to locate a local alignment. Reconstructing
// the alignment path needs the full matrix retained and a second pass; that is a
// deliberate follow-up, not an oversight -- and the WFA experience says the traceback
// is where the bugs live, so it gets its own gate before it is claimed.

#ifndef GENOALIGNER_SW_KERNEL_HIP
#define GENOALIGNER_SW_KERNEL_HIP

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

namespace genoaligner {

// Affine gap penalty. Smith-Waterman needs an explicit scheme: the edit-distance
// unit costs used by WFA are a special case, not the general one.
struct SWParams {
    int match;          // score for a_i == b_j       (typically positive)
    int mismatch;       // score for a_i != b_j       (typically negative)
    int gap_open;       // penalty to OPEN a gap      (positive; subtracted)
    int gap_extend;     // penalty to EXTEND a gap    (positive; subtracted)
};

// One pair. Pointers are DEVICE pointers.
struct SWPairView {
    const char* text;        // rows:    i in [0, text_len)
    int         text_len;
    const char* pattern;     // columns: j in [0, pattern_len)
    int         pattern_len;
};

// Where the best local alignment ends, and its score. Both are useful: the score
// alone tells you whether there is a hit, the end coordinate tells you where.
struct SWResult {
    int score;
    int end_i;   // text index of the last aligned cell
    int end_j;   // pattern index
};

// Score only (no traceback), one block per pair.
//
// `workspace` holds two antidiagonals of the H/E/F matrices, each of
// `2 * (min(text_len, pattern_len) + 1)` ints per pair, because the number of cells
// on an antidiagonal is bounded by that. Allocated by the caller: sizing belongs in
// one place (the Fase 6 lesson: a launch that re-derives sizes passes the wrong one).
// `smax` bounds nothing here (SW is not a banded algorithm) but the caller may pass
// a band width in the future; today it is unused and the matrix is full.
__global__ void sw_score_kernel(const SWPairView* __restrict__ pairs,
                                SWParams           params,
                                SWResult* __restrict__ out);

// The largest useful number of threads per block for this kernel. Exposed so the
// launcher does not guess and so a caller can reason about occupancy.
static constexpr int SW_BLOCK = 128;

}  // namespace genoaligner

#endif  // GENOALIGNER_SW_KERNEL_HIP
