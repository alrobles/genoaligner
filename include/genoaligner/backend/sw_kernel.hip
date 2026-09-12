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
// PARALLELISATION: ONE WARP PER PAIR, ROW BY ROW, COLUMNS TILED BY LANE
// -------------------------------------------------------------------
// The textbook GPU approach is an anti-diagonal sweep with one __syncthreads per
// diagonal. This project has the measurement to avoid it: the WFA kernel's per-wavefront
// barrier pattern produced a time that varied 2.5x between launches doing IDENTICAL work
// (docs/RESULTADO_B6_TCUPS.md), because idle threads waiting on barriers get scheduled
// non-deterministically. An anti-diagonal SW kernel has one barrier per anti-diagonal --
// m+n of them per pair -- which is that failure mode, maximised.
//
// This kernel uses NO BARRIERS AT ALL. One warp owns one pair and walks the matrix row
// by row; inside a row the warp's lanes own columns j = lane+1, lane+1+warpSize, ...
// (tiles of warpSize columns). The non-obvious part is the intra-row dependency:
// E[i][j] and H[i][j] need the CURRENT row's H[i][j-1]/E[i][j-1], which live in the
// neighbouring lane and are being computed in the same step -- a serial chain that a
// plain __shfl_up of the in-flight value cannot break, because in lockstep the source
// lane has not produced it yet.
//
// The chain is broken by reformulating E as a prefix scan. Define, for row i,
//
//     c[j] = max(0, H[i-1][j-1] + s_j, F[i][j])      (no left-dependency: the
//              diagonal comes from a shuffle of the PREVIOUS row's register, and
//              F is vertical so each lane carries its own)
//     w[j] = c[j] + ge*j
//     U[j] = max(U[j-1] + d, w[j]),   U[0] = 0,   d = max(0, ge - go)
//
// Then exactly, for every j >= 1,
//
//     E[i][j] = U[j-1] - go - ge*(j-1)
//     H[i][j] = max(c[j], E[i][j])
//
// (Derivation: E[i][j] = max over k<j of (H[k] - go - ge*(j-1-k)); writing
//  u[k] = H[k] + ge*k gives U[j] = max_{k<=j} u[k], and the affine-gap algebra
//  folds u[j] back into the scan so U[j] = max(U[j-1]+d, c[j]+ge*j). Verified
//  against the direct recurrence on 6000 random cases incl. ge > go.)
//
// Each scan element applies U -> max(U + d, w); composing two adjacent elements is
// associative, (d_p,m_p) then (d_m,m_m) == (d_p+d_m, max(m_p+d_m, m_m)), so a
// Hillis-Steele scan over the warp's lanes computes every prefix in log2(warpSize)
// shuffle steps. Carries between tiles travel by broadcast from the last lane.
//
// What this buys, stated plainly:
//   * blockDim > 1 is CORRECT -- the old grid-stride row walk read columns written
//     by other threads with no way to make them visible (documented defect).
//   * NO __syncthreads and NO __syncwarp: lanes only ever touch shared slots they
//     own; every cross-lane value moves through a register shuffle. Timing is
//     deterministic by construction, not by measurement luck.
//   * Warp-width portable: lane/column mapping and the scan loop both use the
//     runtime warpSize, so the same body runs on 32-lane NVIDIA and 64-lane AMD
//     warps. The CPU gate emulates BOTH widths.
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

// On ROCm >= 5.7 the CUDA-style *_sync warp builtins exist but are opt-in:
// amd_warp_sync_functions.h only declares them when this macro is defined
// before hip_runtime.h is included. __HIP_PLATFORM_AMD__ is itself defined by
// the HIP headers (too late to test here); __HIPCC__ is the driver-defined
// macro available at this point. On the NVIDIA path the amd_* headers are not
// included at all, so the macro is a no-op there; the CPU shim compiles with a
// plain host compiler (no __HIPCC__) and provides the names itself.
#if defined(__HIPCC__) && !defined(HIP_ENABLE_WARP_SYNC_BUILTINS)
#  define HIP_ENABLE_WARP_SYNC_BUILTINS
#endif

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

// Full-warp shuffle mask, portable across the two backends: the _sync intrinsics take
// `unsigned int` on CUDA and `unsigned long long` on ROCm/HIP for AMD. All lanes of a
// warp are always active in this kernel, so the mask is all-ones either way.
#if defined(GENOALIGNER_HIP_SHIM)
#  define GA_WARP_FULL_MASK (~0ull)
#elif defined(__HIP_PLATFORM_NVIDIA__)
#  define GA_WARP_FULL_MASK (0xffffffffu)
#else
#  define GA_WARP_FULL_MASK (~0ull)
#endif

// Sentinel for lanes whose column is beyond the pattern (invalid tiles). Must stay
// representable after adding a few tiles' worth of scan decay, so far from INT_MIN.
static constexpr int SW_NEG_INF = -(1 << 29);

// Warps per block. Each warp owns one pair; the block exists only to pack warps.
// Fixed, not derived from the data: a block sized from the input is how the WFA
// kernel ended up with variable idle occupancy and non-deterministic timing.
static constexpr int SW_WARPS_PER_BLOCK = 4;

// ---- launch contract (one place; the Fase 6 lesson is that a launch site that
// ---- re-derives buffer sizes passes the wrong one) ------------------------
//
//   blockDim.x = SW_WARPS_PER_BLOCK * device warpSize   (32 on NVIDIA, 64 on AMD)
//   gridDim.x  = ceil(n_pairs / SW_WARPS_PER_BLOCK)
//   dyn smem   = SW_WARPS_PER_BLOCK * sw_smem_ints_per_warp(max_pattern_len) * sizeof(int)
//   arg smem_ints_per_warp = sw_smem_ints_per_warp(max_pattern_len)
//
// where max_pattern_len is the largest pattern_len IN THE BATCH. Per-warp shared
// usage is 2*(n+1) ints (previous H row and the F array); E needs no storage --
// it lives inside the scan.
inline int sw_smem_ints_per_warp(int max_pattern_len) { return 2 * (max_pattern_len + 1); }

// Score only (no traceback), one warp per pair.
__global__ void sw_score_kernel(const SWPairView* __restrict__ pairs,
                                SWParams           params,
                                SWResult* __restrict__ out,
                                int                n_pairs,
                                int                smem_ints_per_warp);

}  // namespace genoaligner

#endif  // GENOALIGNER_SW_KERNEL_HIP
