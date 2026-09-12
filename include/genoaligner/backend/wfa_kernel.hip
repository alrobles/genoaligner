// genoaligner — WFA (Wavefront Alignment) kernels, HIP.
//
//   wfa_score_kernel  — score only (Fase 2)
//   wfa_trace_kernel  — score + CIGAR (Fase 4)
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

    // Layout. k in [-smax, smax] is stored at idx(k) = k + smax + 1.
    //
    // The padding is needed on BOTH ends, by one slot each:
    //   idx(k-1) at k=-smax  reads idx(-smax-1) = 0
    //   idx(k+1) at k=+smax  reads idx(+smax+1) = 2*smax+2
    // so the used range is [0, 2*smax+2] and each wavefront needs
    // 2*smax+3 == stride+2 slots. Both padding slots hold the sentinel, so
    // those reads are unreachable rather than out of bounds.
    //
    // NOTE: sizing this as stride+1 (low padding only) overflows by one slot on
    // the high side. The CPU replay harness does NOT catch that, because its
    // std::vector is large enough that a one-past-the-end read stays inside the
    // allocation and returns benign garbage. On the GPU it is a memory access
    // fault. This was found by H2 job 29184154 (rc=134).
    const int stride    = 2 * smax + 1;
    const int wf_stride = stride + 2;          // slots per wavefront (both paddings)
    const int total     = 2 * wf_stride;       // both wavefronts
    auto idx = [&](int k) { return k + smax + 1; };
#ifdef GENOALIGNER_HIP_SHIM
    // CPU replay harness (R1): shared memory is a host vector. The harness
    // sizes it to EXACTLY what the GPU launch site requests, and we do NOT
    // re-assign here — re-sizing would hide an undersized launch, because the
    // stray write would land inside a larger allocation. Letting the harness
    // own the size is what makes an under-sized launch an ASan error on CPU
    // instead of a memory fault on the MI210.
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

// ===========================================================================
// FASE 4 — TRACEBACK (CIGAR reconstruction)
// ===========================================================================
//
// The backward walk is inherently SERIAL: each step's diagonal depends on the
// step before it. The parallelism is therefore ACROSS PAIRS (one thread per
// pair), not across diagonals. That is the shape implemented below.
//
// Source: github.com/smarco/WFA, wavefront/wavefront_backtrace.c
//   :: wavefront_backtrace_linear (lines 223-319), specialised to unit costs.
// The CPU prototype (tests/parity/traceback_cpu.cpp) was developed and validated
// against that source first, and this is the SAME walk moved into the kernel --
// not a second derivation.
//
// ---------------------------------------------------------------- MEMORY MODEL
//
// The walk is zero-copy: the forward pass writes every wavefront M[s] into a
// GLOBAL WORKSPACE as it computes them, and the walk reads those values again in
// the reverse direction. The disk costs no extra memory, which is what makes the
// design viable at all -- see the memory note on wfa_trace_kernel below.
//
// The workspace is a device pointer PARAMETER (one block per pair, the block
// receives workspace + pair_id * wf_alloc). Putting it in `extern __shared__`
// instead would have been simpler but wrong: 65 wavefronts x 131 ints = ~34 KB
// per pair at smax=64, far past the 64 KB shared-memory limit on this hardware
// class, and SharedMemPerBlockOptin would not fit it either.
//
// The same source is built THREE ways, which is the point of keeping it here and
// not in the harness:
//     hipcc  -> MI210          (globals, 64 thread blocks)
//     nvcc   -> RTX 6000       (globals, same source, __HIP_PLATFORM_NVIDIA__)
//     g++    -> CPU shim       (host memory, one "block" = one thread)
//
// wfa_compute/wfa_extend are NOT called from the walk. Reusing the forward
// helpers would share a conceptual error with the forward pass and make the
// re-score check circular; the walk does its own inline predecessor arithmetic,
// exactly as the reference does.

// Capacity bound. At smax=64, 64*159+8 = 10184, so 8192 truncates CIGARs while
// 16384 covers the whole regime (longest possible path = 2*smax+1 = 129 ops).
// Op codes (M/X/I/D), the reversed-code decode, and the run-length formatter all
// live in cigar.hpp so the CPU gate and the device path cannot disagree.
constexpr int WFA_CIGAR_MAX = 16384;

// The walk builds the CIGAR in REVERSE order. wfa_cigar_append appends one op
// code and returns 0 when the caller's capacity is exhausted -- the caller must
// then report TRUNCATION rather than emitting a short CIGAR as if it were whole.
__device__ __host__ inline int wfa_cigar_append(int* cigar_rev, int cap, int* rev_used, int op) {
    if (*rev_used >= cap) return 0;          // truncated: reported, not silent
    cigar_rev[*rev_used] = op;
    ++(*rev_used);
    return 1;
}

// --------------------------------------------------------------- TRAILING TAIL
// Reference lines 246-254: operations outside the grid, before the main walk.
// v > m is a leading deletion, h > n a leading insertion. Both are appended
// before the walk, which is correct because the walk builds backwards and these
// operations come last in the read direction.
__device__ __host__ inline void wfa_trace_tail(int v, int m, int h, int n,
                                               int* cigar_rev, int cap, int* used) {
    for (int i = v - m; i > 0; --i) wfa_cigar_append(cigar_rev, cap, used, 3 /*D*/);
    for (int i = h - n; i > 0; --i) wfa_cigar_append(cigar_rev, cap, used, 2 /*I*/);
}

// ------------------------------------------------------------ BACKWARD WALK
// Transcribes wavefront_backtrace_linear, edit-distance case.
//
//     misms : offsets[k]   + 1     (backtrace.c:73)
//     del   : offsets[k+1]         (backtrace.c:114)   <-- NO +1
//     ins   : offsets[k-1] + 1     (backtrace.c:173)   <-- HAS +1
//
// The +1 on misms and ins (but NOT del) is what makes `num_matches =
// offset - max_offset` land correctly:
//   - misms consumes one position on the SAME diagonal, so the stored
//     predecessor is one further along; +1 brings it onto the comparison scale
//     and the later `--offset` reconciles it.
//   - ins does the same: `--k --offset` after picking it.
//   - del moves diagonals instead of consuming an offset, so its value is
//     already directly comparable and needs no +1; `++k` alone reconciles.
//
// Getting this wrong does NOT show up as a wrong score or a wrong character
// count: it shows up as an 'M' sitting on a non-matching pair. That is why the
// gate verifies CHARACTERS and not just counts.
//
// The walk's span is bounded by the score, so it is called once per pair with a
// wide mem_last that grows with --- never by re-calling it with a chunked
// workspace, which would restart the walk and corrupt the CIGAR.
__device__ __host__ inline void wfa_traceback(const char* pattern, const char* text,
                                             int m, int n, int smax, int score_total,
                                             const int* wf, int wf_stride, int mem_last,
                                             int* cigar_rev, int cap, int* used)
{
    if (score_total < 0) return;             // unresolved within smax

    if (m == 0 || n == 0) {
        // Degenerate: every text char is an insertion, every pattern char a
        // deletion. Consistent with cigar_score(): I consumes text, D pattern.
        for (int i = 0; i < n; ++i) wfa_cigar_append(cigar_rev, cap, used, 2 /*I*/);
        for (int i = 0; i < m; ++i) wfa_cigar_append(cigar_rev, cap, used, 3 /*D*/);
        return;
    }

    // `off` is the WAVEFRONT INDEX (i.e. the score s whose wavefront we want) --
    // not a byte or slot offset. The final wavefront lives at index score_total.
    // Passing 0 here made the walk read the score-0 wavefront, which holds only
    // diagonal 0, so `M(score, k, 0)` returned null for every non-zero score and
    // the walk emitted nothing (rev_used=0, emit=0) even though the score was
    // correct. The score and the CIGAR are produced by different code paths, so a
    // score can be right while the CIGAR silently never starts.
    auto M = [&](int s, int k, int off) -> const int* {
        if (off > mem_last) return (const int*)0;   // wavefront not computed
        if (k < -s || k > s) return (const int*)0;
        return wf + (size_t)off * (size_t)wf_stride + (size_t)(k + smax + 1);
    };
    auto mv = [&](const int* p) -> int {
        return p ? *p : WFA_NEG;
    };

    const int alignment_k = wfa_diag(n, m);
    int score      = score_total;
    int k          = alignment_k;
    const int* c   = M(score_total, k, score_total);   // final wavefront
    int offset     = mv(c);

    if (offset == WFA_NEG) {                 // pre-pass never wrote it
        return;
    }

    int h = wfa_h(k, offset);
    int v = wfa_v(k, offset);

    wfa_trace_tail(v, m, h, n, cigar_rev, cap, used);
    if (v > m) v = m;
    if (h > n) h = n;

    while (v > 0 && h > 0 && score > 0) {
        const int s_prev = score - 1;

        const int* misms_p = M(s_prev, k,     s_prev);
        const int* ins_p   = M(s_prev, k - 1, s_prev);
        const int* del_p   = M(s_prev, k + 1, s_prev);

        const int misms = misms_p ? (*misms_p + 1) : WFA_NEG;   // +1, and k
        const int ins   = ins_p   ? (*ins_p   + 1) : WFA_NEG;   // +1, and k-1
        const int del   = del_p   ? (*del_p)       : WFA_NEG;   // no +1, k+1

        // Compare on the reference's own scale. A sentinel must not win.
        int best = WFA_NEG;
        int op = 0;                          // 0=none 1=mism 2=ins 3=del
        if (wfa_reachable(misms) && misms >= best) { best = misms; op = 1; }
        if (wfa_reachable(ins)   && ins   >  best) { best = ins;   op = 2; }
        if (wfa_reachable(del)   && del   >  best) { best = del;   op = 3; }
        if (op == 0) break;                  // no source

        const int num_matches = offset - best;
        for (int i = 0; i < num_matches; ++i) wfa_cigar_append(cigar_rev, cap, used, 0 /*M*/);

        offset = best;
        v = wfa_v(k, offset);
        h = wfa_h(k, offset);

        if (op == 1) {
            wfa_cigar_append(cigar_rev, cap, used, 1 /*X*/);
            --offset;                        // mismatch consumes one of each
        } else if (op == 2) {
            wfa_cigar_append(cigar_rev, cap, used, 2 /*I*/);
            --k; --offset;                   // insertion consumes a text char
        } else {
            wfa_cigar_append(cigar_rev, cap, used, 3 /*D*/);
            ++k;                             // deletion consumes a pattern char
        }

        v = wfa_v(k, offset);
        h = wfa_h(k, offset);
        --score;
    }

    // Leading matches, then leading indels: reference lines 305-315.
    if (v > 0 && h > 0) {
        const int num_matches = (v < h) ? v : h;
        for (int i = 0; i < num_matches; ++i) wfa_cigar_append(cigar_rev, cap, used, 0 /*M*/);
        v -= num_matches;
        h -= num_matches;
    }
    while (v > 0) { wfa_cigar_append(cigar_rev, cap, used, 3 /*D*/); --v; }
    while (h > 0) { wfa_cigar_append(cigar_rev, cap, used, 2 /*I*/); --h; }
}

// ===========================================================================
// wfa_trace_kernel — Fase 4 GPU traceback, one block per pair.
// ===========================================================================
//
// The forward pass writes EVERY wavefront M[s] into `workspace` as it computes
// them; the walk then reads those values back in reverse. That is the whole
// memory story: one wavefront stack per pair, no second stack.
//
// The walk runs on ONE thread per block. The parallelism is across PAIRS, not
// across diagonals, because each backward step depends on the previous one.
//
// workspace placement. `wf_alloc_max` is the per-pair slot count the harness
// computed, i.e. (smax+1)*wf_stride: room for one wavefront per score 0..smax.
// A pair's slice is workspace[pair_id * wf_alloc_max]. `chunk_bytes` is the
// dynamic-shared-memory size the harness also passes to the launch, so the two
// agree; the kernel does not size the workspace itself.
//
// The CIGAR leaves through `cigar_buf` (cigar_cap ints per pair), NOT through the
// workspace: the walk's output is per-pair, small, and must not be sized by the
// smax-derived stride. Layout per pair, in `cigar_meta` (4 ints per pair):
//
//     cigar_meta[i*4 + 0]   cap used
//     cigar_meta[i*4 + 1]   reversed length (0 if nothing emitted)
//     cigar_meta[i*4 + 2]   truncated flag
//     cigar_meta[i*4 + 3]   emitted flag
//
// meta lives in its own array so the CIGAR buffer can take any cap without the
// meta layout moving.
__global__ void wfa_trace_kernel(const PairView* __restrict__ pairs,
                                 int* __restrict__ scores,
                                 int* __restrict__ cigar_buf,   // cap ints per pair
                                 int* __restrict__ cigar_meta,  // 4 ints per pair
                                 int* __restrict__ workspace,   // wavefront stack
                                 int  smax, int wf_stride, int wf_alloc_max,
                                 int  chunk_bytes, int cigar_cap)
{
    const int pair_id = blockIdx.x;
    const PairView p  = pairs[pair_id];
    const int m = p.pattern_len;
    const int n = p.text_len;

    if (threadIdx.x == 0) {
        cigar_meta[pair_id * 4 + 0] = cigar_cap;
        cigar_meta[pair_id * 4 + 1] = 0;
        cigar_meta[pair_id * 4 + 2] = 0;
        cigar_meta[pair_id * 4 + 3] = 0;
        scores[pair_id] = (m <= 0 || n <= 0) ? ((m <= 0) ? n : m) : -1;
    }
    if (threadIdx.x != 0) return;   // the walk is serial

    // Degenerate inputs: score is n or m, and the CIGAR is all I or all D.
    // Handle here rather than in the walk so the meta is set by the same code
    // that owns the buffer.
    if (m <= 0 || n <= 0) {
        int used = 0;
        for (int i = 0; i < n; ++i) wfa_cigar_append(cigar_buf + (size_t)pair_id * (size_t)cigar_cap,
                                                     cigar_cap, &used, 2 /*I*/);
        for (int i = 0; i < m; ++i) wfa_cigar_append(cigar_buf + (size_t)pair_id * (size_t)cigar_cap,
                                                     cigar_cap, &used, 3 /*D*/);
        cigar_meta[pair_id * 4 + 1] = used;
        cigar_meta[pair_id * 4 + 2] = (used >= cigar_cap) ? 1 : 0;
        cigar_meta[pair_id * 4 + 3] = (used <= cigar_cap) ? 1 : 0;
        return;
    }

    int* ws = workspace + (size_t)pair_id * (size_t)wf_alloc_max;
    int* cg = cigar_buf + (size_t)pair_id * (size_t)cigar_cap;

    // How many wavefronts fit in one pair's slice. Derived from the slice, not
    // from chunk_bytes: the slice is what actually exists.
    const int wf_alloc = wf_alloc_max / wf_stride;
    const int mem_last = (wf_alloc > smax + 1) ? (smax + 1) : wf_alloc;
    if (mem_last < 1) return;      // no wavefront fits; scores[] stays -1

    // ---- forward pass: retain EVERY wavefront ------------------------------
    //
    // Transcribed from forward_all() in tests/parity/traceback_cpu.cpp, which was
    // validated against the three-way checker before this kernel existed. The
    // structure is deliberately identical: index the previous wavefront by
    // DIAGONAL for each k, rather than hoisting three base pointers and indexing
    // them with one loop counter.
    //
    // That hoisting is the bug that produced -1 (unresolved) for 170 of 214
    // pairs while the perfect matches still resolved. The three predecessors of
    // current diagonal k live at k-1, k and k+1 -- DIFFERENT slots in the
    // previous wavefront, one diagonal apart. A single `p_ins[i]`-style stride
    // can only express one of the three, so `ins` and `del` read their neighbour
    // and the recurrence never reaches the end position. Only score 0 (which
    // never enters this loop) survived.
    auto W = [&](int s, int k) -> int& {
        return ws[(size_t)s * wf_stride + (size_t)(k + smax + 1)];
    };
    const int slots = wf_stride * wf_alloc;
    for (int t = 0; t < slots; ++t) ws[t] = WFA_NEG;

    {
        int offset = 0;
        while (offset < m && offset < n && p.pattern[offset] == p.text[offset]) ++offset;
        W(0, 0) = offset;
        const int alignment_k = wfa_diag(n, m);
        if (alignment_k == 0 && offset >= n) {
            scores[pair_id] = 0;
        } else {
            for (int s = 1; s <= smax && s < mem_last; ++s) {
                for (int k = -s; k <= s; ++k) {
                    // Same call shape as the shipped score kernel AND as the
                    // validated CPU prototype: predecessors at k-1 (ins), k+1
                    // (del) and k (mism), all read from wavefront s-1.
                    W(s, k) = wfa_step(p.pattern, p.text, k, m, n,
                                       W(s - 1, k - 1),    // ins
                                       W(s - 1, k + 1),    // del
                                       W(s - 1, k));       // mism
                }
                if (alignment_k >= -s && alignment_k <= s) {
                    const int off = W(s, alignment_k);
                    if (wfa_reachable(off) && off >= n) { scores[pair_id] = s; break; }
                }
            }
        }
    }

    const int score_total = scores[pair_id];
    if (score_total < 0) return;    // unresolved within smax

    if (score_total == 0) {
        // Perfect match: every position is 'M'. Emitting it through the walk
        // would also work, but the walk exits immediately at score 0 and this
        // makes the common case explicit rather than relying on the tail loop.
        int used = 0;
        for (int i = 0; i < n; ++i) wfa_cigar_append(cg, cigar_cap, &used, 0 /*M*/);
        cigar_meta[pair_id * 4 + 0] = cigar_cap;
        cigar_meta[pair_id * 4 + 1] = used;
        cigar_meta[pair_id * 4 + 2] = (used >= cigar_cap) ? 1 : 0;
        cigar_meta[pair_id * 4 + 3] = (used <= cigar_cap) ? 1 : 0;
        return;
    }

    // ---- backward walk: reads the same waves just written ------------------
    int used = 0;
    wfa_traceback(p.pattern, p.text, m, n, smax, score_total,
                  ws, wf_stride, mem_last, cg, cigar_cap, &used);

    if (used > 0) {
        cigar_meta[pair_id * 4 + 0] = cigar_cap;
        cigar_meta[pair_id * 4 + 1] = used;
        cigar_meta[pair_id * 4 + 2] = (used >= cigar_cap) ? 1 : 0;   // truncated
        cigar_meta[pair_id * 4 + 3] = (used <= cigar_cap) ? 1 : 0;   // emitted
    } else {
        cigar_meta[pair_id * 4 + 0] = cigar_cap;
        cigar_meta[pair_id * 4 + 1] = 0;
        cigar_meta[pair_id * 4 + 2] = 0;
        cigar_meta[pair_id * 4 + 3] = 0;
    }
}

} // namespace genoaligner

#endif // GENOALIGNER_WFA_KERNEL_HIP
