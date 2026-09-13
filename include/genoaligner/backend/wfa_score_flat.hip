// genoaligner — Fase 7: score kernel variant with a FIXED block size.
//
// WHY THIS IS A SEPARATE FILE
// ---------------------------
// The validated kernel (wfa_kernel.hip) launches blockDim = 2*smax+1 threads. That
// is the correct COVERAGE -- wavefront s touches diagonals k in [-s, s], so 2*smax+1
// threads can cover any of them -- but it is a terrible OCCUPANCY: the kernel
// returns at s == distance, so with a real distance d the walked wavefronts need at
// most 2*d+1 lanes, and d << smax in the phylogenetic regime the project targets.
//
// Measured (job 29210725 + bench/h7_exit_probe.cpp, 2026-09-11), kernel time vs
// smax at CONSTANT mean distance 15.2:
//
//     smax= 32  block= 128  warps= 4  ->  0.966 ms
//     smax= 64  block= 256  warps= 8  ->  2.190 ms
//     smax=128  block= 512  warps=16  ->  4.835 ms
//     smax=256  block=1024  warps=32  -> 10.349 ms
//
// Time tracks WARP COUNT, not work: the replays show the kernel walks exactly
// ~distance wavefronts at every smax (wf/dist = 1.00), so it is not doing more
// work -- it is paying __syncthreads() over 76-97% idle lanes. The loop length is
// already minimal; the BLOCK SIZE is the defect.
//
// THE CHANGE
// ----------
// Decouple blockDim from smax. A fixed block sweeps the diagonals of each wavefront
// with an interior grid-stride loop:
//
//     for (int t = threadIdx.x; t <= 2*s; t += blockDim.x)  k = t - s
//
// The algebra, the memory layout, the sentinel handling and the termination test
// are ALL UNCHANGED -- this is the same recurrence over the same slots, with the
// threads reassigned. That is deliberate: a layout change would invalidate the
// parity evidence, and there is no reason to risk it when the defect is occupancy.
//
// SOUNDNESS CAVEAT, stated up front: the interior loop lets one thread write
// SEVERAL slots of `cur` before the barrier. The barrier after the loop is
// therefore still required (readers of cur[] must wait for all writers), and the
// number of barriers is unchanged -- only their cost falls, because fewer warps
// are being synchronised. Correctness does not depend on which thread wrote which
// slot, because no slot is read until after the barrier.
//
// This file is included by the parity harness so both kernels are exercised against
// the same control set and the same oracles. It is NOT a replacement until the
// numbers and the parity both say so.

#ifndef GENOALIGNER_WFA_SCORE_FLAT_HIP
#define GENOALIGNER_WFA_SCORE_FLAT_HIP

namespace genoaligner {

// Blocks in this file assume wfa_kernel.hip has already been included (it carries
// PairView, the sentinel, wfa_step, wfa_reachable). Guarded so the include order
// is explicit at the call site rather than accidental.

// Default fixed block. 128 = 4 warps: enough lanes to cover the small wavefronts
// that dominate when d is small, few enough that the barrier is cheap. Tunable,
// and measured rather than assumed -- see the sweep in scripts/h7_flat.sbatch.
#ifndef WFA_FLAT_BLOCK
#define WFA_FLAT_BLOCK 128
#endif

__global__ void wfa_score_kernel_flat(const PairView* __restrict__ pairs,
                                      int* __restrict__ scores,
                                      int  smax)
{
    const int pair_id = blockIdx.x;
    const PairView p  = pairs[pair_id];

    const int m = p.pattern_len;   // rows
    const int n = p.text_len;      // cols

    if (threadIdx.x == 0) {
        if (m <= 0 || n <= 0) scores[pair_id] = (m <= 0) ? n : m;
        else                  scores[pair_id] = -1;
    }
    if (m <= 0 || n <= 0) return;

    // Layout identical to the validated kernel: idx(k) = k + smax + 1, padding on
    // both ends, 2*(2*smax+3) slots shared between the two ping-pong wavefronts.
    const int stride    = 2 * smax + 1;
    const int wf_stride = stride + 2;
    auto idx = [&](int k) { return k + smax + 1; };
#ifdef GENOALIGNER_HIP_SHIM
    int* smem = shim_smem_ptr();
#else
    extern __shared__ int smem[];
#endif
    int* A = smem;
    int* B = smem + wf_stride;

    for (int t = threadIdx.x; t < wf_stride; t += blockDim.x) { A[t] = WFA_NEG; B[t] = WFA_NEG; }
    __syncthreads();

    if (threadIdx.x == 0) {
        int offset = 0;
        while (offset < m && offset < n && p.pattern[offset] == p.text[offset]) ++offset;
        A[idx(0)] = offset;
    }
    __syncthreads();

    const int alignment_k = wfa_diag(n, m);

    if (threadIdx.x == 0) {
        if (alignment_k == 0 && A[idx(0)] >= n) { scores[pair_id] = 0; return; }
    }

    int* prev = A;
    int* cur  = B;

    for (int s = 1; s <= smax; ++s) {
        // Diagonals of this wavefront are k in [-s, s], i.e. slots t = 0..2s.
        // THE ONLY STRUCTURAL CHANGE: a fixed block walks them in grid-stride
        // instead of one-thread-per-slot. hmm_done marks whether THIS thread wrote
        // anything, so an idle thread can skip the (cheap) string compares below --
        // it still must hit the barriers.
        for (int t = threadIdx.x; t <= 2 * s; t += blockDim.x) {
            const int k = t - s;
            cur[idx(k)] = wfa_step(p.pattern, p.text, k, m, n,
                                   prev[idx(k - 1)],    // ins
                                   prev[idx(k + 1)],    // del
                                   prev[idx(k)]);       // mism
        }
        __syncthreads();

        // termination: the single thread owning the alignment diagonal checks it.
        // Same test, same slot, same threshold helper as the validated kernel.
        if (threadIdx.x == 0) {
            if (alignment_k >= -s && alignment_k <= s) {
                const int off = cur[idx(alignment_k)];
                if (wfa_reachable(off) && off >= n) { scores[pair_id] = s; return; }
            }
        }
        __syncthreads();

        int* tmp = prev; prev = cur; cur = tmp;
    }

    return;   // unresolved within smax; scores[] stays -1
}

}  // namespace genoaligner

#endif // GENOALIGNER_WFA_SCORE_FLAT_HIP
