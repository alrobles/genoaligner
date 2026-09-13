// genoaligner — MSA profile-vs-profile trace kernel, body.
//
// One thread per pair: serial forward pass recording direction bytes, then a
// serial backward walk. Deliberately a DIFFERENT code path from
// src/msa/msa_ref.cpp (which re-derives predecessors by float comparison):
// parity between the two is what the gate asserts.
//
// Scratch layout per pair (scratch_stride floats): [prevM | Ix | prevIy],
// each (B.len+1) wide. Ix is the vertical state (consumes A, one array);
// Iy is horizontal (consumed within the row, kept in registers) -- the
// previous row's Iy must also be kept for the M diagonal, hence prevIy.

#ifndef GENOALIGNER_MSA_PP_KERNEL_IMPL_HIP
#define GENOALIGNER_MSA_PP_KERNEL_IMPL_HIP

#include "genoaligner/backend/msa_pp_kernel.hip"

namespace genoaligner {

// Position-specific gap penalties -- must match genomsa::psgp_open/ext in
// msa.hpp exactly (identical float ops, same branch order).
__device__ __host__ inline float pp_open(float occ, const MsaPPParams& p) {
    if (!p.psgp || occ >= 1.0f) return p.gap_open * occ;
    const float q  = p.psgp_scale * p.gap_open * occ;
    const float fl = p.psgp_min_open * p.gap_open;
    return q > fl ? q : fl;
}
__device__ __host__ inline float pp_ext(float occ, const MsaPPParams& p) {
    if (!p.psgp || occ >= 1.0f) return p.gap_extend;
    const float q  = p.gap_extend * occ;
    const float fl = p.psgp_min_ext * p.gap_extend;
    return q > fl ? q : fl;
}

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
                                    int n_pairs, int cigar_cap)
{
    const int pair_id = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (pair_id >= n_pairs) return;

    const MsaPPPair pp = pairs[pair_id];
    const int M = pp.A.len, N = pp.B.len;
    const bool free_ends = (params.free_end_gaps != 0);
    const float ge = params.gap_extend;
    const float NEG = -1e30f;

    if (M <= 0 || N <= 0) {
        out[pair_id] = MsaPPResult{0.f, 0, 0, 0, 0};
        cigar_meta[pair_id * 2 + 0] = cigar_cap;
        cigar_meta[pair_id * 2 + 1] = 0;
        return;
    }
    if ((size_t)(M + 1) * (size_t)(N + 1) > dir_avail[pair_id] ||
        (size_t)(3 * (N + 1) + 2 * (M + 1)) > scr_avail[pair_id]) {
        out[pair_id] = MsaPPResult{MSA_PP_TOO_BIG, 0, 0, 0, 0};
        cigar_meta[pair_id * 2 + 0] = cigar_cap;
        cigar_meta[pair_id * 2 + 1] = 0;
        return;
    }

    uint8_t* dir  = dirs + dir_base[pair_id];
    float* prevM  = scratch + scr_base[pair_id];
    float* Ix     = prevM + (N + 1);
    float* prevIy = Ix + (N + 1);
    float* colM   = prevIy + (N + 1);             // M[i][N] snapshots, i=0..M
    float* colIx  = colM + (M + 1);               // Ix[i][N] snapshots
    const int nrow = N + 1;                       // dir row stride

    // ---- boundary row i = 0 -------------------------------------------------
    prevM[0] = 0.f;
    Ix[0]    = NEG;                               // nothing of A to consume
    prevIy[0] = NEG;
    for (int j = 1; j <= N; ++j) {
        prevM[j]  = free_ends ? 0.f : NEG;
        Ix[j]     = NEG;
        prevIy[j] = free_ends ? 0.f
                              : -(pp_open(pp.B.occ[j - 1], params) + (j - 1) * ge);
    }

    // Snapshot the i=0 boundary column so the endpoint scan can run in the
    // reference's exact 4-phase order after the sweep.
    colM[0] = prevM[N]; colIx[0] = Ix[N];

    // ---- forward sweep -----------------------------------------------------
    for (int i = 1; i <= M; ++i) {
        // values of cell (i,0)
        float curM0  = free_ends ? 0.f : NEG;
        float curIx0 = free_ends ? 0.f
                                 : -(pp_open(pp.A.occ[i - 1], params) + (i - 1) * ge);
        // diagonal predecessors start as the (i-1,0) cell's states
        float dM  = prevM[0];
        float dIx = Ix[0];
        float dIy = prevIy[0];
        prevM[0]  = curM0;
        Ix[0]     = curIx0;
        prevIy[0] = NEG;                          // Iy[i][0]: nothing consumed

        float curMprev = curM0;                   // M[i][j-1]
        float curIyprev = NEG;                    // Iy[i][j-1]
        const float openB = pp_open(pp.A.occ[i - 1], params); // gap in B
        const float extB  = pp_ext (pp.A.occ[i - 1], params);

        for (int j = 1; j <= N; ++j) {
            const float sM = prevM[j], sIx = Ix[j], sIy = prevIy[j]; // (i-1,j)
            // M[i][j]: diagonal predecessor (i-1,j-1), ties M > Ix > Iy
            float mx = dM; uint8_t h = MSA_H_M;
            if (dIx > mx) { mx = dIx; h = MSA_H_IX; }
            if (dIy > mx) { mx = dIy; h = MSA_H_IY; }
            const float curM = msa_col_score(pp.A, i - 1, pp.B, j - 1, params) + mx;
            // Ix[i][j]: consume A_i-1 vs gap in B (vertical state)
            const float oIx = sM - openB;
            const float eIx = sIx - extB;
            const float nIx = (oIx >= eIx) ? oIx : eIx;
            const uint8_t xext = (eIx > oIx) ? MSA_X_EXT : 0;
            // Iy[i][j]: consume B_j-1 vs gap in A (horizontal state)
            const float openA = pp_open(pp.B.occ[j - 1], params);
            const float oIy = curMprev - openA;
            const float eIy = curIyprev - pp_ext(pp.B.occ[j - 1], params);
            const float cIy = (oIy >= eIy) ? oIy : eIy;
            const uint8_t yext = (eIy > oIy) ? MSA_Y_EXT : 0;

            dir[i * nrow + j] = h | xext | yext;
            // shift diagonals and store this row's states
            dM = sM; dIx = sIx; dIy = sIy;
            prevM[j] = curM; Ix[j] = nIx; prevIy[j] = cIy;
            curMprev = curM; curIyprev = cIy;
        }
        colM[i] = prevM[N]; colIx[i] = Ix[N];
    }
    // ---- endpoint scan, exact reference order ------------------------------
    //   column N: per-i, mM then mIx       (i ascending)
    //   row M   : per-j, mM then mIy       (j ascending)
    // Strict '>' everywhere: the first maximum found wins.
    float best = NEG; int ei = M, ej = N; char estate = 'M';
    if (free_ends) {
        for (int i = 0; i <= M; ++i) {
            if (colM[i]  > best) { best = colM[i];  ei = i; ej = N; estate = 'M'; }
            if (colIx[i] > best) { best = colIx[i]; ei = i; ej = N; estate = 'I'; }
        }
        for (int j = 0; j <= N; ++j) {
            if (prevM[j]  > best) { best = prevM[j];  ei = M; ej = j; estate = 'M'; }
            if (prevIy[j] > best) { best = prevIy[j]; ei = M; ej = j; estate = 'D'; }
        }
    } else {
        best = prevM[N]; ei = M; ej = N; estate = 'M';
    }

    // ---- backward walk -----------------------------------------------------
    // Emits ops into the pair's cigar slice in WALK order (reversed relative
    // to the alignment); the caller reverses the used range.
    uint8_t* cig = cigar_buf + cig_base[pair_id];
    int used = 0;
    int i = ei, j = ej; char st = estate;
    while ((i > 0 || j > 0) && used < cigar_cap) {
        if (st == 'M') {
            if (i == 0 || j == 0) break;
            uint8_t b = dir[i * nrow + j];
            cig[used++] = 0;                      // 'M'
            uint8_t h = b & 3;
            --i; --j;
            st = (h == MSA_H_M) ? 'M' : (h == MSA_H_IX ? 'I' : 'D');
        } else if (st == 'I') {
            if (i == 0) break;
            cig[used++] = 1;                      // 'I' = A column only
            if (j > 0) st = (dir[i * nrow + j] & MSA_X_EXT) ? 'I' : 'M';
            --i;
        } else {                                  // 'D'
            if (j == 0) break;
            cig[used++] = 2;                      // 'D' = B column only
            if (i > 0) st = (dir[i * nrow + j] & MSA_Y_EXT) ? 'D' : 'M';
            --j;
        }
    }
    cigar_meta[pair_id * 2 + 0] = cigar_cap;
    cigar_meta[pair_id * 2 + 1] = used;
    out[pair_id] = MsaPPResult{best, i, j, ei, ej};
}

// ---------------------------------------------------------------------------
// V2 KERNEL: one BLOCK per pair. Threads sweep the anti-diagonals in
// parallel; a cell (i,j) on diagonal d=i+j reads only diagonals d-1 and d-2,
// so a block barrier between diagonals is the entire synchronisation. The
// recurrence, the direction bytes, the endpoint-scan order and the walk are
// IDENTICAL to msa_pp_trace_kernel -- only the traversal order changed, and
// traversal order cannot change any value (every read is of a strictly older
// diagonal). Parity with the serial kernel and the reference is still what
// the gate asserts.
//
// Launch: <<<n_pairs, T>>>; pair p is block p. T must be a multiple of the
// warp size (256 works on gfx and NVIDIA).
//
// Scratch layout per pair (floats), all rolling arrays indexed by ROW i:
//   [M3 | Ix3 | Iy3]   3 slots each (d%3), (M+1) floats per slot
//   [colM | colIx]     (i,N) snapshots for the endpoint scan, M+1 each
//   [rowM | rowIy]     (M,j) snapshots, N+1 each
//   need = 11*(M+1) + 2*(N+1) floats.
// ---------------------------------------------------------------------------
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
                                       int n_pairs, int cigar_cap)
{
    const int pair_id = blockIdx.x;
    if (pair_id >= n_pairs) return;

    const MsaPPPair pp = pairs[pair_id];
    const int M = pp.A.len, N = pp.B.len;
    const bool free_ends = (params.free_end_gaps != 0);
    const float ge = params.gap_extend;
    const float NEG = -1e30f;

    if (M <= 0 || N <= 0) {
        if (threadIdx.x == 0) {
            out[pair_id] = MsaPPResult{0.f, 0, 0, 0, 0};
            cigar_meta[pair_id * 2 + 0] = cigar_cap;
            cigar_meta[pair_id * 2 + 1] = 0;
        }
        return;
    }
    if ((size_t)(M + 1) * (size_t)(N + 1) > dir_avail[pair_id] ||
        (size_t)(11 * (M + 1) + 2 * (N + 1)) > scr_avail[pair_id]) {
        if (threadIdx.x == 0) {
            out[pair_id] = MsaPPResult{MSA_PP_TOO_BIG, 0, 0, 0, 0};
            cigar_meta[pair_id * 2 + 0] = cigar_cap;
            cigar_meta[pair_id * 2 + 1] = 0;
        }
        return;
    }

    uint8_t* dir   = dirs + dir_base[pair_id];
    float*   M3    = scratch + scr_base[pair_id];
    float*   Ix3   = M3  + 3 * (M + 1);
    float*   Iy3   = Ix3 + 3 * (M + 1);
    float*   colM  = Iy3 + 3 * (M + 1);
    float*   colIx = colM + (M + 1);
    float*   rowM  = colIx + (M + 1);
    float*   rowIy = rowM + (N + 1);
    const int nrow = N + 1;
    const int tid = (int)threadIdx.x, T = (int)blockDim.x;
    const int w = M + 1;

    for (int d = 0; d <= M + N; ++d) {
        const int lo = (d > N) ? d - N : 0;
        const int hi = (d < M) ? d : M;
        const int cs = d % 3, p1 = (d + 2) % 3, p2 = (d + 1) % 3;
        float*       cM = M3 + cs * w,  *cIx = Ix3 + cs * w,  *cIy = Iy3 + cs * w;
        const float* qM = M3 + p2 * w,  *qIx = Ix3 + p2 * w,  *qIy = Iy3 + p2 * w;
        const float* sM = M3 + p1 * w,  *sIx = Ix3 + p1 * w;
        const float* tM = M3 + p1 * w,  *tIy = Iy3 + p1 * w;
        for (int i = lo + tid; i <= hi; i += T) {
            const int j = d - i;
            float vM, vIx, vIy;
            if (i == 0) {
                vM  = (j == 0 || free_ends) ? 0.f : NEG;
                vIx = NEG;
                vIy = (j == 0) ? NEG
                     : (free_ends ? 0.f : -(pp_open(pp.B.occ[j - 1], params) + (j - 1) * ge));
            } else if (j == 0) {
                vM  = free_ends ? 0.f : NEG;
                vIx = free_ends ? 0.f : -(pp_open(pp.A.occ[i - 1], params) + (i - 1) * ge);
                vIy = NEG;
            } else {
                // M[i][j]: diagonal predecessor (i-1,j-1) on d-2, row i-1.
                float mx = qM[i - 1]; uint8_t h = MSA_H_M;
                if (qIx[i - 1] > mx) { mx = qIx[i - 1]; h = MSA_H_IX; }
                if (qIy[i - 1] > mx) { mx = qIy[i - 1]; h = MSA_H_IY; }
                vM = msa_col_score(pp.A, i - 1, pp.B, j - 1, params) + mx;
                // Ix[i][j]: (i-1,j) on d-1, row i-1.
                const float oIx = sM[i - 1] - pp_open(pp.A.occ[i - 1], params);
                const float eIx = sIx[i - 1] - pp_ext(pp.A.occ[i - 1], params);
                vIx = (oIx >= eIx) ? oIx : eIx;
                const uint8_t xext = (eIx > oIx) ? MSA_X_EXT : 0;
                // Iy[i][j]: (i,j-1) on d-1, row i.
                const float oIy = tM[i] - pp_open(pp.B.occ[j - 1], params);
                const float eIy = tIy[i] - pp_ext(pp.B.occ[j - 1], params);
                vIy = (oIy >= eIy) ? oIy : eIy;
                const uint8_t yext = (eIy > oIy) ? MSA_Y_EXT : 0;
                dir[i * nrow + j] = h | xext | yext;
            }
            cM[i] = vM; cIx[i] = vIx; cIy[i] = vIy;
            if (j == N) { colM[i] = vM; colIx[i] = vIx; }
            if (i == M) { rowM[j] = vM; rowIy[j] = vIy; }
        }
        __syncthreads();
    }

    if (tid != 0) return;

    // ---- endpoint scan, exact reference order (single thread) --------------
    float best = NEG; int ei = M, ej = N; char estate = 'M';
    if (free_ends) {
        for (int i = 0; i <= M; ++i) {
            if (colM[i]  > best) { best = colM[i];  ei = i; ej = N; estate = 'M'; }
            if (colIx[i] > best) { best = colIx[i]; ei = i; ej = N; estate = 'I'; }
        }
        for (int j = 0; j <= N; ++j) {
            if (rowM[j]  > best) { best = rowM[j];  ei = M; ej = j; estate = 'M'; }
            if (rowIy[j] > best) { best = rowIy[j]; ei = M; ej = j; estate = 'D'; }
        }
    } else {
        best = rowM[N]; ei = M; ej = N; estate = 'M';
    }

    // ---- backward walk (single thread, identical to serial kernel) ---------
    uint8_t* cig = cigar_buf + cig_base[pair_id];
    int used = 0;
    int i = ei, j = ej; char st = estate;
    while ((i > 0 || j > 0) && used < cigar_cap) {
        if (st == 'M') {
            if (i == 0 || j == 0) break;
            uint8_t b = dir[i * nrow + j];
            cig[used++] = 0;
            uint8_t h = b & 3;
            --i; --j;
            st = (h == MSA_H_M) ? 'M' : (h == MSA_H_IX ? 'I' : 'D');
        } else if (st == 'I') {
            if (i == 0) break;
            cig[used++] = 1;
            if (j > 0) st = (dir[i * nrow + j] & MSA_X_EXT) ? 'I' : 'M';
            --i;
        } else {
            if (j == 0) break;
            cig[used++] = 2;
            if (i > 0) st = (dir[i * nrow + j] & MSA_Y_EXT) ? 'D' : 'M';
            --j;
        }
    }
    cigar_meta[pair_id * 2 + 0] = cigar_cap;
    cigar_meta[pair_id * 2 + 1] = used;
    out[pair_id] = MsaPPResult{best, i, j, ei, ej};
}

}  // namespace genoaligner

#endif  // GENOALIGNER_MSA_PP_KERNEL_IMPL_HIP
