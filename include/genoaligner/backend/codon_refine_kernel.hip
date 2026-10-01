// genoaligner — codon stage-2 phase-DP kernel ("genomsa --codon --refine").
//
// Semantics are FIXED by codon_realign_row in src/msa/msa_ref.cpp: the
// frozen-footprint phase DP (one state per (codon slot k, raw-nt start s)
// with |s - 3k| <= band). The DP body in codon_refine_kernel_impl.hip is
// compiled BOTH by the device kernel and by the host path -- identical
// source, not a re-derivation -- so parity between the two is structural
// (the CPU shim compiles it too). This header is host-TU safe: with no HIP
// toolchain/shim in sight the kernel qualifiers degrade to no-ops.
//
// One thread per row: serial forward pass recording a direction byte per
// cell (the chosen step+1, 0 = unreached), then the HOST walks it back
// (traceback needs std::string ops and emits variable-length inserts).
// Kernel outputs therefore are: tr bytes + the endpoint s per row.
//
// Direction byte: step+1 where step in {3,4,2,5,1,0} preference order
//   1 = in-frame (st 3), 2 = fs ins +1 (st 4), 3 = fs del -1 (st 2),
//   4 = fs ins +2 (st 5), 5 = fs del -2 (st 1), 6 = empty slot (st 0).
//   0 marks unreached cells -- the host traceback stops on it.
//
// Packed buffers (per row r):
//   seq_base[r] -> seqs : raw nt chars, len seq_len[r]
//   stok[r*C+j]         : row's token at backbone column j (-1 absent)
//   pi_base[r]  -> pi   : footprint, pi_len[r] column indices
//   tr_base[r]  -> tr   : K*ND direction bytes (ND = 2*band+1)
//   scr_base[r] -> scr  : 2*ND floats of rolling dp rows
// Tables (shared, read-only):
//   dot[t*C+j] = col_score of token t vs column j over ALL rows
//   ocnt[j]    = non-gap row count; ncnt[j*12+p*4+b] = base-b count at
//                codon position p; sub = 65x65 codon matrix.

#ifndef GENOALIGNER_CODON_REFINE_KERNEL_HIP
#define GENOALIGNER_CODON_REFINE_KERNEL_HIP

#include <cstddef>
#include <cstdint>

#if defined(GENOALIGNER_HIP_SHIM) || defined(__HIPCC__) || \
    defined(__CUDACC__) || defined(__HIP_DEVICE_COMPILE__) || \
    defined(__CUDA_ARCH__)
#include <hip/hip_runtime.h>
#else
#define __global__
#define __device__
#define __host__
#endif

// Non-contractible float ops (same contract as GA_F*_RN in
// msa_pp_kernel.hip; a distinct prefix so both headers can coexist).
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#define CR_FMUL_RN(a, b) __fmul_rn((a), (b))
#define CR_FADD_RN(a, b) __fadd_rn((a), (b))
#else
#define CR_FMUL_RN(a, b) ((a) * (b))
#define CR_FADD_RN(a, b) ((a) + (b))
#endif

namespace genoaligner {

struct CodonRefineParams {
    float match, ts, tv;          // nt match/ts/tv for marginals
    float gap_open;               // base for psgp_open
    int   psgp;                   // 1 = position-specific gap penalties
    float psgp_scale, psgp_min_open;
    float fs, fs_term;            // frameshift event costs (internal/terminal)
    int   band;                   // D: max |s - 3k| drift, nt
    int   nseq;                   // inv_nm1 = 1/(nseq-1)
    int   C;                      // backbone codon columns
};

__global__ void codon_refine_kernel(const char* __restrict__ seqs,
                                  const size_t* __restrict__ seq_base,
                                  const int* __restrict__ seq_len,
                                  const int8_t* __restrict__ stok,
                                  const int* __restrict__ pi,
                                  const size_t* __restrict__ pi_base,
                                  const int* __restrict__ pi_len,
                                  const float* __restrict__ dot,
                                  const float* __restrict__ ocnt,
                                  const float* __restrict__ ncnt,
                                  const float* __restrict__ sub,
                                  CodonRefineParams params,
                                  uint8_t* __restrict__ tr,
                                  const size_t* __restrict__ tr_base,
                                  float* __restrict__ scr,
                                  const size_t* __restrict__ scr_base,
                                  int* __restrict__ bes,
                                  int n_rows);

}  // namespace genoaligner

#endif  // GENOALIGNER_CODON_REFINE_KERNEL_HIP
