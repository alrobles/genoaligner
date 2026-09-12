// genoaligner — Smith-Waterman score kernel, body.
//
// Included after the declaration in sw_kernel.hip. Kept separate so the CPU shim can
// execute this body on host memory (tests/parity/hip_cpu_shim), which is how every
// kernel in this project is gated in seconds without a GPU.

#ifndef GENOALIGNER_SW_KERNEL_IMPL_HIP
#define GENOALIGNER_SW_KERNEL_IMPL_HIP

#include "genoaligner/backend/sw_kernel.hip"

namespace genoaligner {

// Row-major walk with the previous H row in shared memory.
//
// Thread t owns columns j = t+1, t+1+blockDim, ... of every row (grid-stride inside the
// row). Column 0 is the boundary and is written by thread 0 only. One __syncthreads per
// ROW: see the header for why not one per anti-diagonal.
__global__ void sw_score_kernel(const SWPairView* __restrict__ pairs,
                                SWParams           params,
                                SWResult* __restrict__ out)
{
    const int pair_id = blockIdx.x;
    const SWPairView p = pairs[pair_id];

    const int m = p.text_len;      // rows
    const int n = p.pattern_len;   // columns

    // Empty side: no local alignment exists. Report score 0 with no position.
    if (m <= 0 || n <= 0) {
        if (threadIdx.x == 0) out[pair_id] = SWResult{0, -1, -1};
        return;
    }

    // H_prev[j]  : H of row i-1 for column j      (n+1 entries, index 0 = boundary)
    // E_prev[j]  : E of row i   for column j      (gap in TEXT, consumes pattern)
    // F_prev[j]  : F of row i-1 for column j      (gap in PATTERN, consumes text)
#ifdef GENOALIGNER_HIP_SHIM
    // CPU replay harness: shared memory is a host vector the harness sizes to exactly
    // what the launch site would request. Same arrangement as the WFA kernel, and for
    // the same reason: re-sizing here would hide an undersized launch, because a stray
    // write would land inside a larger allocation and go unnoticed.
    int* smem = shim_smem_ptr();
#else
    extern __shared__ int smem[];
#endif
    int* H = smem;              // (n+1)
    int* E = H + (n + 1);       // (n+1)
    int* F = E + (n + 1);       // (n+1)

    for (int j = threadIdx.x; j <= n; j += blockDim.x) {
        H[j] = 0; E[j] = 0; F[j] = 0;
    }

    // Best-so-far, reduced across the block at the end of each row. Kept per-thread so
    // no barrier is needed to update it, then combined once.
    int  my_best = 0;
    int  my_bi = -1;
    int  my_bj = -1;

    __syncthreads();

    for (int i = 1; i <= m; ++i) {
        const char a = p.text[i - 1];

        // diag_prev carries H[i-1][j-1] as the row walk advances, which is why the
        // columns must be visited in increasing order within a thread's stride. Each
        // thread keeps its own copy for its own columns, so no barrier is needed
        // between columns -- only between rows.
        int diag_prev = 0;   // H[i-1][0] == 0

        for (int j = (int)threadIdx.x + 1; j <= n; j += (int)blockDim.x) {
            const char b = p.pattern[j - 1];
            const int  s = (a == b) ? params.match : params.mismatch;

            // E: gap in text, opening from H[i][j-1] or extending E[i][j-1].
            // Threads stride by blockDim, so H[j-1] and E[j-1] may belong to ANOTHER
            // thread of the same row. That is why E and H are shared and why the row
            // is completed before moving on -- within a row, the values a thread reads
            // at j-1 were written earlier in the same row (by the previous thread for
            // consecutive columns, or by this thread across rows for its own stride).
            int e = E[j - 1] - params.gap_extend;
            const int e_open = H[j - 1] - params.gap_open;
            if (e_open > e) e = e_open;

            // F: gap in pattern, opening from H[i-1][j] or extending F[i-1][j].
            int f = F[j] - params.gap_extend;
            const int f_open = H[j] - params.gap_open;
            if (f_open > f) f = f_open;

            int h = diag_prev + s;
            if (e > h) h = e;
            if (f > h) h = f;
            if (h < 0) h = 0;                 // LOCAL: restart instead of going negative

            // Update the thread's own running maximum BEFORE overwriting diag_prev.
            if (h > my_best) { my_best = h; my_bi = i - 1; my_bj = j - 1; }

            // Shift for the next column in this thread's stride, then store.
            const int h_diag_next = H[j];     // H[i-1][j] becomes the diagonal for j+1
            H[j] = h;
            E[j] = e;
            F[j] = f;
            diag_prev = h_diag_next;
        }
        __syncthreads();
    }

    // Reduce the per-thread maxima. Uses shared memory with a simple tree; the counts
    // are small (blockDim) so this is negligible next to the DP itself.
    __shared__ int r_best[SW_BLOCK];
    __shared__ int r_i[SW_BLOCK];
    __shared__ int r_j[SW_BLOCK];
    r_best[threadIdx.x] = my_best;
    r_i[threadIdx.x]    = my_bi;
    r_j[threadIdx.x]    = my_bj;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            if (r_best[threadIdx.x + stride] > r_best[threadIdx.x]) {
                r_best[threadIdx.x] = r_best[threadIdx.x + stride];
                r_i[threadIdx.x]    = r_i[threadIdx.x + stride];
                r_j[threadIdx.x]    = r_j[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) out[pair_id] = SWResult{r_best[0], r_i[0], r_j[0]};
}

}  // namespace genoaligner

#endif  // GENOALIGNER_SW_KERNEL_IMPL_HIP
