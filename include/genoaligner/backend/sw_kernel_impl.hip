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

// ===========================================================================
// FASE B — traceback
// ===========================================================================
//
// The walk is serial and so is this kernel's forward pass: one thread owns one
// pair end to end. That is a deliberate different code path from sw_score_kernel's
// warp scan -- if the two disagree, at least one is wrong, which is the property
// the parity gate needs. Speed of traceback is Fase D's question, not this one's.

// Same shape as wfa_cigar_append: append one op to the REVERSED cigar array;
// returns 0 when the caller's cap is exhausted so truncation is reported, never
// silently emitted short.
__device__ __host__ inline int sw_cigar_append(int* rev, int cap, int* used, int op)
{
    if (*used >= cap) return 0;
    rev[*used] = op;
    ++(*used);
    return 1;
}

// Backward walk over the direction table. bi/bj are the 1-based matrix
// coordinates of the best cell (the alignment's last cell); o_start_* come back
// as the 0-based index of the first aligned character in each sequence.
//
// The walk's states: 0 = on H, 1 = inside an E gap (text-gap, emitting 'D'),
// 2 = inside an F gap (pattern-gap, emitting 'I'). In a gap state the byte read
// at the CURRENT cell says how that cell's E/F value was formed: extend keeps the
// state, open returns to H -- the open's consumed column was already emitted.
__device__ __host__ inline void sw_trace_walk(const char* text, const char* pattern,
                                              const uint8_t* dirs, int row_stride,
                                              int bi, int bj,
                                              int* o_start_i, int* o_start_j,
                                              int* cigar_rev, int cap, int* used,
                                              int* ok)
{
    int i = bi, j = bj, state = 0;
    *ok = 1;
    while (i > 0 && j > 0) {
        const uint8_t b = dirs[(size_t)i * (size_t)row_stride + (size_t)j];
        if (state == 0) {
            const uint8_t hsrc = b & 3;
            if (hsrc == SW_DIR_STOP) break;
            if (hsrc == SW_DIR_DIAG) {
                *ok &= sw_cigar_append(cigar_rev, cap, used,
                        (text[i - 1] == pattern[j - 1]) ? SW_OP_M : SW_OP_X);
                --i; --j;
            } else if (hsrc == SW_DIR_E) {
                // H[i][j] = E[i][j]: column j is a gap op. Where the path lands
                // next depends on how E[i][j] ITSELF was formed: an open ends the
                // gap (predecessor H[i][j-1]), an extend continues it. Getting
                // this from esrc -- not landing in E unconditionally -- is what
                // keeps a length-1 gap from being emitted as two D's.
                *ok &= sw_cigar_append(cigar_rev, cap, used, SW_OP_D);
                state = (b & SW_DIR_EEXT) ? 1 : 0;
                --j;
            } else {                       // SW_DIR_F
                *ok &= sw_cigar_append(cigar_rev, cap, used, SW_OP_I);
                state = (b & SW_DIR_FEXT) ? 2 : 0;
                --i;
            }
        } else if (state == 1) {           // in an E gap at cell (i,j)
            *ok &= sw_cigar_append(cigar_rev, cap, used, SW_OP_D);
            --j;
            if (!(b & SW_DIR_EEXT)) state = 0;
        } else {                           // in an F gap at cell (i,j)
            *ok &= sw_cigar_append(cigar_rev, cap, used, SW_OP_I);
            --i;
            if (!(b & SW_DIR_FEXT)) state = 0;
        }
    }
    *o_start_i = i;                        // 0-based first aligned char = prefix left
    *o_start_j = j;
}

// sw_trace_kernel — score + CIGAR, one thread per pair.
//
// Forward pass: the plain serial recurrence, the SAME arithmetic as the CPU
// reference (tests/sw/test_sw_reference.cpp) plus the direction record. The
// boundary cells (row 0, column 0) are written as STOP bytes; E[.][0] = 0 and
// F[0][.] = 0 match the reference's zero-initialised edges exactly, so a gap can
// never "extend" in from outside the matrix with a different value than the
// reference would compute.
__global__ void sw_trace_kernel(const SWPairView* __restrict__ pairs,
                                SWParams           params,
                                SWTraceResult* __restrict__ out,
                                int* __restrict__ cigar_buf,
                                int* __restrict__ cigar_meta,
                                uint8_t* __restrict__ dirs,    size_t dir_stride,
                                int* __restrict__ scratch,     size_t scratch_stride,
                                int n_pairs, int cigar_cap)
{
    const int pair_id = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (pair_id >= n_pairs) return;

    int* meta = cigar_meta + (size_t)pair_id * 4;
    meta[0] = cigar_cap; meta[1] = 0; meta[2] = 0; meta[3] = 0;
    out[pair_id] = SWTraceResult{0, -1, -1, -1, -1};

    const SWPairView p = pairs[pair_id];
    const int m = p.text_len, n = p.pattern_len;
    if (m <= 0 || n <= 0) {
        meta[3] = 1;                       // empty alignment IS the result
        return;
    }

    const int rs = n + 1;                  // direction-table row stride (cells)
    if ((size_t)(m + 1) * (size_t)rs > dir_stride ||
        2 * (size_t)rs > scratch_stride) {
        out[pair_id].score = SW_TRACE_TOO_BIG;   // declared limit, no writes
        return;
    }

    uint8_t* d     = dirs    + (size_t)pair_id * dir_stride;
    int*     Hprev = scratch + (size_t)pair_id * scratch_stride;
    int*     Farr  = Hprev + rs;

    for (int j = 0; j <= n; ++j) { Hprev[j] = 0; Farr[j] = 0; d[j] = SW_DIR_STOP; }
    for (int i = 0; i <= m; ++i) d[(size_t)i * rs] = SW_DIR_STOP;

    const int go = params.gap_open, ge = params.gap_extend;
    int best = 0, bi = -1, bj = -1;        // matrix coords, 1-based

    for (int i = 1; i <= m; ++i) {
        const char a = p.text[i - 1];
        int hleft = 0;                     // H[i][j-1]
        int e_row = 0;                     // E[i][j-1]  (E[i][0] = 0, as the reference)
        int diag  = 0;                     // H[i-1][j-1]; column 0 is always 0
        for (int j = 1; j <= n; ++j) {
            // Identical arithmetic to the reference: candidates first, then the
            // max with strictly-greater wins, so ties resolve diag > E > F and a
            // gap extends only when it beats opening. The reference walk the gate
            // compares against breaks ties the same way, so paths must be EQUAL.
            int e = hleft - go;
            uint8_t esrc = 0;
            { const int e2 = e_row - ge; if (e2 > e) { e = e2; esrc = SW_DIR_EEXT; } }
            int f = Hprev[j] - go;
            uint8_t fsrc = 0;
            { const int f2 = Farr[j] - ge; if (f2 > f) { f = f2; fsrc = SW_DIR_FEXT; } }
            int h = diag + ((a == p.pattern[j - 1]) ? params.match : params.mismatch);
            uint8_t hsrc = SW_DIR_DIAG;
            if (e > h) { h = e; hsrc = SW_DIR_E; }
            if (f > h) { h = f; hsrc = SW_DIR_F; }
            if (h <= 0) { h = 0; hsrc = SW_DIR_STOP; }   // h==0 restarts: stop, not a 0-move
            d[(size_t)i * rs + j] = (uint8_t)(hsrc | esrc | fsrc);

            diag     = Hprev[j];           // becomes H[i-1][j] for the next column
            Hprev[j] = h;
            Farr[j]  = f;
            hleft    = h;
            e_row    = e;
            if (h > best) { best = h; bi = i; bj = j; }
        }
    }

    if (best > 0) {
        int used = 0, si = bi, sj = bj, ok = 1;
        sw_trace_walk(p.text, p.pattern, d, rs, bi, bj, &si, &sj,
                      cigar_buf + (size_t)pair_id * (size_t)cigar_cap,
                      cigar_cap, &used, &ok);
        out[pair_id] = SWTraceResult{best, si, sj, bi - 1, bj - 1};
        meta[1] = used;
        meta[2] = ok ? 0 : 1;              // ran out of cap mid-walk
        meta[3] = ok ? 1 : 0;
    } else {
        meta[3] = 1;                       // score 0: the empty CIGAR is correct
    }
}

}  // namespace genoaligner

#endif  // GENOALIGNER_SW_KERNEL_IMPL_HIP
