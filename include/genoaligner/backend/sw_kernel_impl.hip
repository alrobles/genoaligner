// genoaligner — Smith-Waterman score kernel, body.
//
// Included after the declaration in sw_kernel.hip. Kept separate so the CPU shim can
// execute this body on host memory (tests/parity/hip_cpu_shim), which is how every
// kernel in this project is gated in seconds without a GPU. For this kernel the shim
// matters even more than usual: shim::run_block executes it with REAL threads and
// REAL exchanges, so the blockDim>1 behaviour below is what the parity gate checks --
// the previous row-walk version could only be verified at blockDim=1 precisely
// because the shim could not emulate cross-lane dataflow. Now it can.
//
// One warp per pair, row by row, lanes owning tiled columns. The header file has the
// scan algebra that makes the intra-row dependency parallel-safe, and the reason
// there are no barriers anywhere in this body.

#ifndef GENOALIGNER_SW_KERNEL_IMPL_HIP
#define GENOALIGNER_SW_KERNEL_IMPL_HIP

#include "genoaligner/backend/sw_kernel.hip"

namespace genoaligner {

__global__ void sw_score_kernel(const SWPairView* __restrict__ pairs,
                                SWParams           params,
                                SWResult* __restrict__ out,
                                int                n_pairs,
                                int                smem_ints_per_warp)
{
    const int wsize = warpSize;                      // 32 NVIDIA / 64 AMD; settable under the shim
    const int lane  = (int)threadIdx.x % wsize;
    const int wid   = (int)threadIdx.x / wsize;
    const int npw   = (int)blockDim.x / wsize;       // warps packed into this block

    // The contract (sw_kernel.hip) is blockDim = warps * warpSize. On a malformed
    // launch the truncated last warp would map onto pairs the next block also owns;
    // refuse it instead of computing the same pair twice.
    if (wid >= npw) return;

    const int pair_id = (int)blockIdx.x * npw + wid;
    if (pair_id >= n_pairs) return;                  // whole warp exits together

    const SWPairView p = pairs[pair_id];
    const int m = p.text_len;      // rows
    const int n = p.pattern_len;   // columns

    // Empty side: no local alignment exists. Report score 0 with no position.
    // The exit is uniform across the warp, so no lane is left at a shuffle alone.
    if (m <= 0 || n <= 0) {
        if (lane == 0) out[pair_id] = SWResult{0, -1, -1};
        return;
    }

    // Shared memory. Per warp: a private slice of smem_ints_per_warp ints laid out as
    //     [ Hprev[0..n] | F[0..n] ]      each nstride = smem_ints_per_warp/2 ints wide
    // Hprev[j] = H of row i-1 at column j; F[j] = the vertical-gap value for column j.
    // E gets NO array: it is folded into the row scan (see the header).
    //
    // Every slot is read and written ONLY by the lane that owns that column
    // (j % wsize == lane, counting from 1), so no barrier -- block or warp -- is
    // needed for it. Cross-lane values all move through __shfl_*_sync on registers.
#ifdef GENOALIGNER_HIP_SHIM
    // CPU replay harness: shared memory is a host vector the harness sizes to exactly
    // what the launch site would request. Same arrangement as the WFA kernel, and for
    // the same reason: re-sizing here would hide an undersized launch, because a stray
    // write would land inside a larger allocation and go unnoticed.
    int* smem = shim_smem_ptr();
#else
    extern __shared__ int smem[];
#endif
    const int nstride = smem_ints_per_warp / 2;
    int* Hprev = smem + wid * smem_ints_per_warp;
    int* Farr  = Hprev + nstride;

    for (int j = lane + 1; j <= n; j += wsize) { Hprev[j] = 0; Farr[j] = 0; }

    const int go = params.gap_open;
    const int ge = params.gap_extend;
    // Scan decay per column: d = max(0, ge - go). With the usual go > ge this is 0
    // and the scan degenerates to a plain prefix max; the general form keeps the
    // kernel exact for every parameter choice.
    const int dscan = (ge - go > 0) ? (ge - go) : 0;

    // Best-so-far, per lane; reduced across the warp by shuffles at the end.
    int best = 0, bi = -1, bj = -1;
    const int ntiles = (n + wsize - 1) / wsize;

    for (int i = 1; i <= m; ++i) {
        const char a = p.text[i - 1];

        int U_carry = 0;    // U at column 0: H[i][0] + ge*0 = 0
        int d_carry = 0;    // H[i-1][0] = 0 (boundary column)

        for (int t = 0; t < ntiles; ++t) {
            const int j = t * wsize + lane + 1;
            const bool valid = (j <= n);

            const int h_prev = valid ? Hprev[j] : 0;   // H[i-1][j]
            const int f_prev = valid ? Farr[j]  : 0;   // F[i-1][j]

            // diag = H[i-1][j-1]: the previous lane's h_prev in this tile, or --
            // for lane 0 -- the last lane's h_prev of the PREVIOUS tile, carried.
            // All lanes execute every shuffle: the calls are unconditional so the
            // warp stays converged (and so the emulated exchanges line up).
            int diag = __shfl_up_sync(GA_WARP_FULL_MASK, h_prev, 1);
            if (lane == 0) diag = d_carry;

            // F[i][j] = max(H[i-1][j] - go, F[i-1][j] - ge): vertical, per-column,
            // entirely inside this lane's registers/slots.
            int f_new = h_prev - go;
            { const int e2 = f_prev - ge; if (e2 > f_new) f_new = e2; }

            const int s = valid
                ? ((a == p.pattern[j - 1]) ? params.match : params.mismatch)
                : 0;
            int c = diag + s;
            if (f_new > c) c = f_new;
            if (c < 0)   c = 0;

            // --- the row scan (header algebra) --------------------------------
            // Element at column j: (dscan, w[j]) applies U -> max(U + dscan, w[j]).
            // Invalid lanes contribute (dscan, SW_NEG_INF): their decay can only
            // reach lanes after them, which are invalid too -- the last tile's
            // trailing lanes never feed a valid column nor a used carry.
            int dm = dscan;
            int mm = valid ? c + ge * j : SW_NEG_INF;
            for (int off = 1; off < wsize; off <<= 1) {
                const int dp = __shfl_up_sync(GA_WARP_FULL_MASK, dm, off);
                const int mp = __shfl_up_sync(GA_WARP_FULL_MASK, mm, off);
                if (lane >= off) {
                    // Incoming segment precedes mine: (d_p+d, max(m_p+d, m)).
                    const int thru = mp + dm;
                    if (thru > mm) mm = thru;
                    dm += dp;
                }
            }
            // Apply the tile's composed operator to the incoming carry: this lane's
            // inclusive prefix U[j].
            const int U_incl = (U_carry + dm > mm) ? U_carry + dm : mm;
            int U_excl = __shfl_up_sync(GA_WARP_FULL_MASK, U_incl, 1);
            if (lane == 0) U_excl = U_carry;

            // E[i][j] = U[j-1] - go - ge*(j-1);  H[i][j] = max(c[j], E[i][j]).
            const int e = U_excl - go - ge * (j - 1);
            int h = (c > e) ? c : e;
            if (!valid) h = 0;

            if (valid && h > best) { best = h; bi = i - 1; bj = j - 1; }
            if (valid) { Hprev[j] = h; Farr[j] = f_new; }

            // Carries into the next tile, broadcast from the last lane: U at the
            // tile's final column, and H[i-1] there (next tile's lane-0 diagonal).
            // Unused after the last tile; computed anyway so every lane hits the
            // same shuffle sequence.
            U_carry = __shfl_sync(GA_WARP_FULL_MASK, U_incl, wsize - 1);
            d_carry = __shfl_sync(GA_WARP_FULL_MASK, h_prev, wsize - 1);
        }
    }

    // Warp-wide argmax by xor-butterfly: every lane ends with the global winner;
    // lane 0 reports it. Ties keep the first lane's coordinate -- the score is the
    // quantity under test, not which optimal cell reported it.
    for (int off = wsize / 2; off > 0; off >>= 1) {
        const int ob = __shfl_xor_sync(GA_WARP_FULL_MASK, best, off);
        const int oi = __shfl_xor_sync(GA_WARP_FULL_MASK, bi,   off);
        const int oj = __shfl_xor_sync(GA_WARP_FULL_MASK, bj,   off);
        if (ob > best) { best = ob; bi = oi; bj = oj; }
    }
    if (lane == 0) out[pair_id] = SWResult{best, bi, bj};
}

}  // namespace genoaligner

#endif  // GENOALIGNER_SW_KERNEL_IMPL_HIP
