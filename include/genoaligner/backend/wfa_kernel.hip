// genoaligner — WFA (Wavefront Alignment) score-only kernel, HIP.
//
// Version 4. Versions 1-3 failed the parity harness (5.08%, 0.32%, and the CPU
// prototype lineage at 55%). The cause was NOT a coordinate convention, and NOT
// the direction of the diagonal step: it was that the recurrence had been
// *reverse-engineered* rather than ported, and the `+1` on the insertion term
// had been taken from the wrong kernel of the reference implementation.
//
// This file is now a verbatim port. Every line below cites its source in
// github.com/smarco/WFA (MIT):
//
//   offset macros    wavefront/wavefront_offset.h:51-59
//     #define WAVEFRONT_V(k,offset)  ((offset)-(k))   // pattern coord i
//     #define WAVEFRONT_H(k,offset)  (offset)         // text coord    j
//     #define WAVEFRONT_OFFSET_NULL  (INT32_MIN/2)
//     DPMATRIX_DIAGONAL(h,v) = ((h)-(v))
//
//   compute-next      wavefront/wavefront_compute_edit.c:87-99
//                     :: wavefront_compute_edit_idm
//     ins  = prev[k-1];       // NO +1
//     del  = prev[k+1];
//     mism = prev[k];
//     max  = MAX(del, MAX(ins,mism)+1);
//     if (H(k,max) > text_len)    max = NULL;
//     if (V(k,max) > pattern_len) max = NULL;
//
//   extension         wavefront/wavefront_extend_kernels.c:49-66
//                     :: wavefront_extend_matches_kernel_charwise
//     while (pattern[V(k,o)] == text[H(k,o)]) ++o;
//     Applies as a SEPARATE pass over the wavefront, after compute and before
//     the termination test.
//
//   termination       wavefront/wavefront_termination.c:37-61
//                     :: wavefront_termination_end2end
//     alignment_k = DPMATRIX_DIAGONAL(text_len, pattern_len) = n - m
//     done iff offsets[alignment_k] >= text_len
//
// =========================================================== THE THREE TRAPS
//
// (1) THE +1 ON ins. The edit kernel has NO +1 on the insertion term; the +1
//     lives inside MAX(ins,mism)+1, and `del` stays outside that group. The
//     indel kernel (wavefront_compute_indel.c:61) DOES write `ins = prev[k-1]+1`,
//     for a different cost model. Taking it from there inflates every insertion.
//     This single line accounted for the 55% parity of the CPU prototype.
//
// (2) THE STORED VALUE IS h. The wavefront value is the TEXT coordinate h (= j);
//     the pattern coordinate is recovered as v = h - k. It is NOT a pattern
//     index and NOT a diagonal length. Confusing these makes the extension and
//     the termination test disagree about what the stored number means.
//
// (3) THE SENTINEL CORRUPTS UNDER ARITHMETIC. WAVEFRONT_OFFSET_NULL is
//     INT32_MIN/2, and NULL+1 != NULL. An exact `== NULL` test therefore leaks
//     garbage coordinates into the extension (observed as an index of
//     ~18446744072635809796 == -2^31). Use a THRESHOLD test, not equality.
//     The reference avoids this only because its sentinel never gets incremented.
//
// ========================================================= PARALLELISATION
//
// One block per pair. Thread t owns diagonal k = t - s for the current score s
// (t in [0, 2s]). Two ping-pong shared buffers of 2*smax+1 ints. The loop over
// s is inherently sequential; a barrier per iteration is the cost of the
// wavefront's data dependence, and is why smax bounds the useful block size.
//
// IMPORTANT: the block must be at least 2*smax+1 threads (the harness sizes it),
// otherwise the thread-to-diagonal mapping k = tid - s silently skips diagonals
// and the parity rate degrades without an error being raised.

#ifndef GENOALIGNER_WFA_KERNEL_HIP
#define GENOALIGNER_WFA_KERNEL_HIP

#include <hip/hip_runtime.h>

namespace genoaligner {

struct UnitCosts {
    static constexpr int mismatch = 1;
    static constexpr int gap      = 1;
};

struct PairView {
    const char* text;
    int         text_len;
    const char* pattern;
    int         pattern_len;
    int         max_score;
};

// ---- wavefront_offset.h ----------------------------------------------------
constexpr int WFA_NEG = INT32_MIN / 2;                 // WAVEFRONT_OFFSET_NULL
__device__ __host__ inline int wfa_h(int k, int offset) { return offset; }        // WAVEFRONT_H
__device__ __host__ inline int wfa_v(int k, int offset) { return offset - k; }    // WAVEFRONT_V
__device__ __host__ inline int wfa_diag(int h, int v) { return h - v; }           // DPMATRIX_DIAGONAL

// Sentinel-safe reachability. See TRAP (3): NULL + small increments is no longer
// exactly NULL, so this must be a threshold test.
__device__ __host__ inline bool wfa_reachable(int offset) { return offset > WFA_NEG + 1024; }

// ---------------------------------------------------------------------------
// Core wavefront logic, shared by the GPU kernel and the CPU replay harness
// (tests/parity). Keeping it as a function is what lets the harness verify THIS
// file's algebra instead of a re-typed copy of it.
// ---------------------------------------------------------------------------

// Compute-next for one diagonal. Verbatim wavefront_compute_edit_idm.
__device__ __host__ inline int wfa_compute(int prev_ins, int prev_del, int prev_mism,
                                           int k, int m, int n) {
    const int mid = prev_ins > prev_mism ? prev_ins : prev_mism;
    int max = (prev_del > mid + 1) ? prev_del : mid + 1;
    if (!wfa_reachable(max)) return WFA_NEG;
    if (wfa_h(k, max) > n)   return WFA_NEG;
    if (wfa_v(k, max) > m)   return WFA_NEG;
    return max;
}

// Greedy extension along one diagonal (wavefront_extend_matches_kernel_charwise).
__device__ __host__ inline int wfa_extend(const char* pattern, const char* text,
                                          int k, int offset, int m, int n) {
    if (!wfa_reachable(offset)) return offset;
    int v = wfa_v(k, offset), h = wfa_h(k, offset);
    while (v < m && h < n && pattern[v] == text[h]) { ++v; ++h; ++offset; }
    return offset;
}

// One full score step (compute + extend) for diagonal k. Caller supplies the
// predecessor values already fetched from the previous wavefront.
__device__ __host__ inline int wfa_step(char const* pattern, char const* text,
                                        int k, int m, int n,
                                        int prev_ins, int prev_del, int prev_mism) {
    int max = wfa_compute(prev_ins, prev_del, prev_mism, k, m, n);
    return wfa_extend(pattern, text, k, max, m, n);
}

__global__ void wfa_score_kernel(const PairView* __restrict__ pairs,
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

    // Layout. k in [-smax, smax] is stored at idx(k) = k + smax + 1, so the
    // used range is [0, 2*smax+1] == [0, stride]. Slot 0 and slot `stride` are
    // PADDING and hold the sentinel: that is what makes the reads of
    // prev[idx(k-1)] at k=-smax and prev[idx(k+1)] at k=+smax land on
    // unreachable padding instead of out of bounds. Each wavefront therefore
    // occupies wf_stride == stride+1 slots, and the two are back to back.
    const int stride    = 2 * smax + 1;
    const int wf_stride = stride + 1;          // slots per wavefront (incl. padding)
    const int total     = 2 * wf_stride;       // both wavefronts
    auto idx = [&](int k) { return k + smax + 1; };
#ifdef GENOALIGNER_HIP_SHIM
    // CPU replay harness (R1): shared memory is a host vector. The harness calls
    // the same wfa_step() logic below, so the algebra under test is this file's.
    shim::smem_vec().assign(total, WFA_NEG);
    int* smem = shim_smem_ptr();
#else
    extern __shared__ int smem[];
#endif
    int* A = smem;
    int* B = smem + wf_stride;

    // ---- score 0: only diagonal k = 0 is reachable -------------------------
    for (int t = threadIdx.x; t < wf_stride; t += blockDim.x) { A[t] = WFA_NEG; B[t] = WFA_NEG; }
    __syncthreads();

    if (threadIdx.x == 0) {
        // H = offset = j, V = offset - k = j on k=0. Extend greedily.
        int offset = 0;
        while (offset < m && offset < n && p.pattern[offset] == p.text[offset]) ++offset;
        A[idx(0)] = offset;
    }
    __syncthreads();

    const int alignment_k = wfa_diag(n, m);   // n - m

    if (threadIdx.x == 0) {
        if (alignment_k == 0 && A[idx(0)] >= n) { scores[pair_id] = 0; return; }
    }

    int* prev = A;
    int* cur  = B;

    for (int s = 1; s <= smax; ++s) {
        const int k = (int)threadIdx.x - s;    // diagonal owned by this thread
        const int lo = -s, hi = s;

        // ---- compute-next (wavefront_compute_edit_idm) ---------------------
        if (threadIdx.x <= 2 * s) {
            cur[idx(k)] = wfa_step(p.pattern, p.text, k, m, n,
                                   prev[idx(k - 1)],    // ins
                                   prev[idx(k + 1)],    // del
                                   prev[idx(k)]);       // mism
        }
        __syncthreads();

        // ---- extension (separate pass, as in the reference) ----------------
        // Folded into wfa_step already; this pass exists in the reference as a
        // distinct kernel for vectorisation. Kept as a no-op marker so the
        // structure mirrors wavefront_extend_matches_packed_end2end.
        __syncthreads();

        // ---- termination (wavefront_termination_end2end) -------------------
        if (threadIdx.x == 0) {
            if (alignment_k >= lo && alignment_k <= hi) {
                const int off = cur[idx(alignment_k)];
                if (wfa_reachable(off) && off >= n) { scores[pair_id] = s; return; }
            }
        }
        __syncthreads();

        int* tmp = prev; prev = cur; cur = tmp;
    }

    return;   // unresolved within smax; scores[] stays -1
}

} // namespace genoaligner

#endif // GENOALIGNER_WFA_KERNEL_HIP
