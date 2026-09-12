// genoaligner — Smith-Waterman score kernel implementation (antidiagonal).
//
// This is the body included from sw_kernel.hip's declaration. Kept in the same
// translation-unit family as the WFA kernel so the CPU shim can execute it on host
// memory (tests/parity/hip_cpu_shim), which is how every kernel in this project is
// gated in seconds without a GPU.

#ifndef GENOALIGNER_SW_KERNEL_IMPL_HIP
#define GENOALIGNER_SW_KERNEL_IMPL_HIP

#include "genoaligner/backend/sw_kernel.hip"

namespace genoaligner {

namespace sw_detail {

// Index into a pair's antidiagonal buffers. Antidiagonal d has cells (i, j) with
// i + j == d. There are at most min(m,n)+1 cells on any antidiagonal, so one slot per
// cell index within the diagonal, per matrix, times two diagonals (current, previous).
struct Layout {
    int diag_cells;   // min(text_len, pattern_len) + 1, padded to a multiple of 4
};

static inline int diag_slot(int i, int j) { return (i < j) ? i : j; }

}  // namespace sw_detail

__global__ void sw_score_kernel(const SWPairView* __restrict__ pairs,
                                SWParams           params,
                                SWResult* __restrict__ out)
{
    const int pair_id = blockIdx.x;
    const SWPairView p = pairs[pair_id];

    const int m = p.text_len;      // rows
    const int n = p.pattern_len;   // columns

    // A pair with an empty side has no local alignment to report: score 0 at the
    // origin. Handled explicitly so the loops below never see m or n as 0.
    if (m <= 0 || n <= 0) {
        if (threadIdx.x == 0) out[pair_id] = SWResult{0, -1, -1};
        return;
    }

    // Two diagonals of three matrices (H, E, F). The cell count per diagonal is
    // bounded by min(m, n) + 1; pad so slot arithmetic never collides.
    const int cells = ((m < n ? m : n) + 3) & ~3;
    extern __shared__ int smem[];
    int* Hc = smem;               // current diagonal
    int* Hp = Hc + cells;         // previous diagonal
    int* Ec = Hp + cells;
    int* Ep = Ec + cells;
    int* Fc = Ep + cells;
    int* Fp = Fc + cells;

    // shared memory: 6 * cells ints
    for (int t = threadIdx.x; t < 6 * cells; t += blockDim.x) smem[t] = 0;

    // The best score and its end coordinate, accumulated across diagonals by thread 0
    // from a block-wide reduction.
    __shared__ int s_best;
    __shared__ int s_bi;
    __shared__ int s_bj;
    if (threadIdx.x == 0) { s_best = 0; s_bi = -1; s_bj = -1; }
    __syncthreads();

    const int total_diags = m + n - 1;

    for (int d = 0; d < total_diags; ++d) {
        // Cells of diagonal d: i from max(0, d-(n-1)) to min(m-1, d), j = d - i.
        const int i_lo = (d - (n - 1) > 0) ? (d - (n - 1)) : 0;
        const int i_hi = (d < m - 1) ? d : (m - 1);

        // Grid-stride over the cells of this antidiagonal: fixed block size, so the
        // number of idle threads does not depend on the data (the WFA lesson).
        for (int i = i_lo + (int)threadIdx.x; i <= i_hi; i += (int)blockDim.x) {
            const int j = d - i;
            const int slot = sw_detail::diag_slot(i, j);

            // Diagonal predecessors live at slot-1 or slot depending on which matrix
            // we read, because i and j advance differently along the antidiagonal.
            const int h_diag = (i - 1 >= 0 && j - 1 >= 0)
                             ? sw_detail::diag_slot(i - 1, j - 1) : -1;
            const int h_left = (j - 1 >= 0) ? sw_detail::diag_slot(i, j - 1) : -1;
            const int h_up   = (i - 1 >= 0) ? sw_detail::diag_slot(i - 1, j)   : -1;

            const int prev_h = (h_diag >= 0 && i - 1 + j - 1 == d - 2) ? Hp[h_diag] : 0;
            // E and F for the current cell come from the PREVIOUS diagonal as well.
            const int e_prev = (h_left >= 0) ? Ep[h_left] : 0;
            const int f_prev = (h_up   >= 0) ? Fp[h_up]   : 0;
            const int e_h    = (h_left >= 0) ? Hp[h_left] : 0;
            const int f_h    = (h_up   >= 0) ? Hp[h_up]   : 0;

            const char a = p.text[i];
            const char b = p.pattern[j];
            const int  s = (a == b) ? params.match : params.mismatch;

            int e = e_h - params.gap_open;
            const int e_ext = e_prev - params.gap_extend;
            if (e_ext > e) e = e_ext;

            int f = f_h - params.gap_open;
            const int f_ext = f_prev - params.gap_extend;
            if (f_ext > f) f = f_ext;

            int h = prev_h + s;
            if (e > h) h = e;
            if (f > h) h = f;
            if (h < 0) h = 0;            // LOCAL: restart rather than go negative

            Hc[slot] = h;
            Ec[slot] = e;
            Fc[slot] = f;

            if (h > s_best) { /* recorded after the barrier, from a reduction */ }
        }
        __syncthreads();

        // Reduce this diagonal's maximum and remember the argmax. Done after the
        // barrier so every thread's writes are visible.
        for (int i = i_lo + (int)threadIdx.x; i <= i_hi; i += (int)blockDim.x) {
            const int j = d - i;
            const int h = Hc[sw_detail::diag_slot(i, j)];
            if (threadIdx.x == 0) {
                if (h > s_best) { s_best = h; s_bi = i; s_bj = j; }
            }
        }
        __syncthreads();

        // Ping-pong: the buffers written this iteration become "previous" for the
        // next. Swapping pointers (not copying) is why both diagonals live in the
        // same allocation.
        int* tmp;
        tmp = Hp; Hp = Hc; Hc = tmp;
        tmp = Ep; Ep = Ec; Ec = tmp;
        tmp = Fp; Fp = Fc; Fc = tmp;
        for (int t = threadIdx.x; t < 3 * cells; t += blockDim.x) {
            Hc[t] = 0; Ec[t] = 0; Fc[t] = 0;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) out[pair_id] = SWResult{s_best, s_bi, s_bj};
}

}  // namespace genoaligner

#endif  // GENOALIGNER_SW_KERNEL_IMPL_HIP
