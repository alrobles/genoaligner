// genoaligner — MSA profile-vs-profile semiglobal Gotoh kernel ("genomsa").
//
// Semantics are FIXED by src/msa/msa_ref.cpp -- this kernel must reproduce it
// exactly, not approximate it:
//
//   M[i][j]  = col_score(A_i-1, B_j-1) + max(M[i-1][j-1], Ix[i-1][j-1], Iy[i-1][j-1])
//   Ix[i][j] = max(M[i-1][j] - go*occA[i-1],  Ix[i-1][j] - ge)   (A col vs gap in B)
//   Iy[i][j] = max(M[i][j-1] - go*occB[j-1],  Iy[i][j-1] - ge)   (B col vs gap in A)
//
//   col_score = sum_a sum_b fa*fb*S(a,b)   -- gap fraction never participates
//   free_end_gaps: first row/col = 0; endpoint = best cell on last row/col
//   traceback tie order M > Ix > Iy, recorded once in the forward pass.
//
// Column layout on device is SoA: five float arrays per profile
// (A,C,G,T,gap fractions) plus occupancy. A lane reading column j touches
// five consecutive floats -- coalesced across lanes.
//
// V1 KERNEL: one thread per pair, serial forward pass recording a direction
// byte per cell, then a serial backward walk. It is deliberately a DIFFERENT
// code path from the CPU reference (which re-derives predecessors by value):
// agreement between them is what parity means. The warp-per-pair score kernel
// (M3) is validated against THIS kernel.
//
// Direction byte (same spirit as sw_kernel Fase B):
//   [1:0] hsrc  0=M  1=Ix  2=Iy          how M[i][j] was produced
//   [2]   xsrc  Ix[i][j] extended (1) or opened from M (0)
//   [3]   ysrc  Iy[i][j] extended (1) or opened from M (0)
//
// Workspace per pair: (M+1)*(N+1) direction bytes. A pair that would exceed
// dir_stride reports score = MSA_PP_TOO_BIG -- a declared limit, never an
// out-of-bounds write.

#ifndef GENOALIGNER_MSA_PP_KERNEL_HIP
#define GENOALIGNER_MSA_PP_KERNEL_HIP

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace genoaligner {

// Scoring parameters, float to match the reference exactly.
struct MsaPPParams {
    float match;          // S(a,a)
    float ts;             // transition mismatch  (A<->G, C<->T)
    float tv;             // transversion mismatch
    float gap_open;       // scaled by opposing occupancy
    float gap_extend;
    int   free_end_gaps;  // 1 = semiglobal (fragments), 0 = global
};

// One profile, SoA on device. cols[b] has `len` floats, b in 0..4
// (0=A,1=C,2=G,3=T,4=gap). occ has `len` floats.
struct MsaProfileView {
    const float* cols[5];
    const float* occ;
    int          len;
};

// One pair to align.
struct MsaPPPair {
    MsaProfileView A;
    MsaProfileView B;
};

// Result of the semiglobal profile alignment.
struct MsaPPResult {
    float score;
    int   ai, aj;   // aligned span starts (first consumed column of A/B)
    int   bi, bj;   // aligned span ends (exclusive)
};

// Reported in MsaPPResult::score when the pair's matrix exceeds dir_stride.
static constexpr float MSA_PP_TOO_BIG = -1e30f;

// Direction-byte fields.
static constexpr uint8_t MSA_H_M  = 0;
static constexpr uint8_t MSA_H_IX = 1;
static constexpr uint8_t MSA_H_IY = 2;
static constexpr uint8_t MSA_X_EXT = 4;
static constexpr uint8_t MSA_Y_EXT = 8;

// Column-vs-column weighted sum-of-pairs. Device and host share it through
// the CPU shim: the same body must compile under g++ (tests/parity/hip_cpu_shim).
__device__ __host__ inline float msa_col_score(
        const MsaProfileView& A, int i,
        const MsaProfileView& B, int j,
        const MsaPPParams& p) {
    float s = 0.f;
    for (int a = 0; a < 4; ++a) {
        float fa = A.cols[a][i];
        if (fa == 0.f) continue;
        for (int b = 0; b < 4; ++b) {
            float fb = B.cols[b][j];
            if (fb == 0.f) continue;
            float sub = (a == b) ? p.match
                        : ((a == 0 && b == 2) || (a == 2 && b == 0) ||
                           (a == 1 && b == 3) || (a == 3 && b == 1))
                            ? p.ts : p.tv;
            s += fa * fb * sub;
        }
    }
    return s;
}

// ---- launch contract ---------------------------------------------------------
//
//   pair_id    = blockIdx.x * blockDim.x + threadIdx.x   (one thread per pair)
//   Buffers are PACKED: each pair owns a variable-size slice given by a base
//   offset array, so a level batch mixing 1 kb leaves with 20 kb profiles
//   wastes nothing.
//   dir_base   : n_pairs offsets into dirs (bytes). Pair p needs
//                (A.len+1)*(B.len+1) bytes at dirs[dir_base[p]].
//   scr_base   : n_pairs offsets into scratch (floats). Pair p needs
//                3*(B.len+1) + 2*(A.len+1) floats.
//   cig_base   : n_pairs offsets into cigar_buf (bytes). Pair p needs
//                cigar_cap bytes (ops: 0=M both, 1=A only, 2=B only, stored
//                in WALK order -- the caller reverses the used range).
//   dir_avail  : for each pair, the byte budget at dir_base[p] (n_pairs).
//                A pair whose matrix exceeds it reports MSA_PP_TOO_BIG and
//                emits nothing -- the caller packs tight, this guards the
//                packing arithmetic itself.
//   cigar_meta : n_pairs * 2 ints = {cap, used}.
__global__ void msa_pp_trace_kernel(const MsaPPPair* __restrict__ pairs,
                                    MsaPPParams params,
                                    MsaPPResult* __restrict__ out,
                                    uint8_t* __restrict__ dirs,
                                    const size_t* __restrict__ dir_base,
                                    const size_t* __restrict__ dir_avail,
                                    uint8_t* __restrict__ cigar_buf,
                                    const size_t* __restrict__ cig_base,
                                    int* __restrict__ cigar_meta,
                                    float* __restrict__ scratch,
                                    const size_t* __restrict__ scr_base,
                                    const size_t* __restrict__ scr_avail,
                                    int n_pairs, int cigar_cap);

// V2: same contract, one BLOCK per pair (blockIdx.x = pair id), anti-diagonal
// parallel sweep. Pair p needs 11*(A.len+1) + 2*(B.len+1) scratch floats at
// scr_base[p]. Semantics identical to msa_pp_trace_kernel.
__global__ void msa_pp_trace_kernel_wf(const MsaPPPair* __restrict__ pairs,
                                       MsaPPParams params,
                                       MsaPPResult* __restrict__ out,
                                       uint8_t* __restrict__ dirs,
                                       const size_t* __restrict__ dir_base,
                                       const size_t* __restrict__ dir_avail,
                                       uint8_t* __restrict__ cigar_buf,
                                       const size_t* __restrict__ cig_base,
                                       int* __restrict__ cigar_meta,
                                       float* __restrict__ scratch,
                                       const size_t* __restrict__ scr_base,
                                       const size_t* __restrict__ scr_avail,
                                       int n_pairs, int cigar_cap);

}  // namespace genoaligner

#endif  // GENOALIGNER_MSA_PP_KERNEL_HIP
