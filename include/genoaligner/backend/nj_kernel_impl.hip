// genoaligner — exact dense Neighbor Joining kernels (NJ0, docs/SPEC_NJ_GPU.md).
//
// Three kernels per round, all bit-exact with genomsa::nj_tree:
//   nj_rowsum_kernel   r[a] = sum_{b != a, b in alive order} d[alive[a]][alive[b]]
//                      one thread per a, SEQUENTIAL sum in host order (no atomics)
//   nj_argmin_kernel   per row a: lexicographic min of (q, a, b) over b > a
//   nj_merge_kernel    one block: reduce the m partials, record the merge,
//                      write the new node's row into slot alive[bi], compact
//                      alive (stable, new node appended) -- exactly the host's
//                      `rest + [u]`.
//
// State layout (device):
//   d          n*n double, row-major, symmetric. Slots are leaf ids 0..n-1; the
//              node produced by a merge REUSES the slot of its first child.
//   alive      m slots in host order; double-buffered across rounds.
//   slot_node  slot -> Tree node id (n + round after a merge).
//   merges     2 ints per round: (node(xi), node(xj)); downloaded once at end.
//
// Arithmetic is the host's, operation for operation:
//   q     = (double)(m-2) * d - r[a] - r[b]
//   d_new = 0.5 * (d_iv + d_jv - d_ij)
// The product is done through __dmul_rn on device so no FMA contraction can
// change the last bit of q relative to the (non-fused) x86-64 host build.

#ifndef GENOALIGNER_NJ_KERNEL_IMPL_HIP
#define GENOALIGNER_NJ_KERNEL_IMPL_HIP

#include <hip/hip_runtime.h>
#include <cstddef>

namespace genoaligner {

// Block width of the reduction kernels. Max on device; the CPU shim spawns one
// host thread per lane, so it runs the same code at a narrow block.
#ifdef GENOALIGNER_HIP_SHIM
static constexpr int NJ_T = 32;
#else
static constexpr int NJ_T = 256;
#endif
static constexpr int NJ_T_MAX = 256;
// Rowsum has only m threads of work; a narrow block spreads them over more CUs.
#ifdef GENOALIGNER_HIP_SHIM
static constexpr int NJ_RS_T = 32;
#else
static constexpr int NJ_RS_T = 64;
#endif

struct NjBest { double q; int a, b; };   // a < 0: empty

__device__ inline bool nj_better(const NjBest& x, const NjBest& y) {
    // true if x should replace y: y empty, smaller q, or same q and earlier (a,b)
    if (y.a < 0) return x.a >= 0;
    if (x.a < 0) return false;
    if (x.q != y.q) return x.q < y.q;
    if (x.a != y.a) return x.a < y.a;
    return x.b < y.b;
}

#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#define NJ_UNROLL _Pragma("unroll")
#else
#define NJ_UNROLL
#endif

__device__ inline double nj_mul(double x, double y) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __dmul_rn(x, y);
#else
    return x * y;
#endif
}

// Block-wide lexicographic reduction of sh[0..blockDim.x) into sh[0].
__device__ inline void nj_block_reduce(NjBest* sh) {
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (nj_better(sh[threadIdx.x + s], sh[threadIdx.x]))
                sh[threadIdx.x] = sh[threadIdx.x + s];
        }
        __syncthreads();
    }
}

__global__ void nj_rowsum_kernel(const double* __restrict__ d,
                                 const int* __restrict__ alive,
                                 int m, int n, double* __restrict__ r)
{
    __shared__ int sh_alive[NJ_T_MAX];
    const int a  = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int sa = a < m ? alive[a] : 0;
    double s = 0;
    // d is symmetric: read column sa down the alive rows so neighbouring
    // threads (neighbouring a) touch neighbouring addresses. alive[] is staged
    // through shared memory per tile and the tile body is unrolled so the
    // loads are in flight together; the adds stay strictly in host order.
    constexpr int U = 8;
    for (int b0 = 0; b0 < m; b0 += (int)blockDim.x) {
        __syncthreads();
        if (b0 + (int)threadIdx.x < m) sh_alive[threadIdx.x] = alive[b0 + threadIdx.x];
        __syncthreads();
        if (a >= m) continue;
        const int tile = m - b0 < (int)blockDim.x ? m - b0 : (int)blockDim.x;
        int k = 0;
        for (; k + U <= tile; k += U) {
            double v[U];
NJ_UNROLL
            for (int u = 0; u < U; ++u)
                v[u] = d[(size_t)sh_alive[k + u] * (size_t)n + (size_t)sa];
NJ_UNROLL
            for (int u = 0; u < U; ++u)
                if (b0 + k + u != a) s += v[u];
        }
        for (; k < tile; ++k)
            if (b0 + k != a) s += d[(size_t)sh_alive[k] * (size_t)n + (size_t)sa];
    }
    if (a < m) r[a] = s;
}

// grid = m blocks (one per a), block = NJ_T. part[a] = lexicographic argmin
// over b in (a, m). Rows with no b > a produce an empty partial.
__global__ void nj_argmin_kernel(const double* __restrict__ d,
                                 const int* __restrict__ alive,
                                 const double* __restrict__ r,
                                 int m, int n, NjBest* __restrict__ part)
{
    __shared__ NjBest sh[NJ_T_MAX];
    const int a = (int)blockIdx.x;
    NjBest best{0.0, -1, -1};
    if (a < m - 1) {
        const int    sa = alive[a];
        const double ra = r[a];
        const double mm = (double)(m - 2);
        const double* row = d + (size_t)sa * (size_t)n;
        for (int b = a + 1 + (int)threadIdx.x; b < m; b += (int)blockDim.x) {
            const double q = nj_mul(mm, row[alive[b]]) - ra - r[b];
            const NjBest c{q, a, b};
            if (nj_better(c, best)) best = c;
        }
    }
    sh[threadIdx.x] = best;
    __syncthreads();
    nj_block_reduce(sh);
    if (threadIdx.x == 0) part[a] = sh[0];
}

// grid = 1 block, block = NJ_T.
__global__ void nj_merge_kernel(double* __restrict__ d,
                                const int* __restrict__ alive_in,
                                int* __restrict__ alive_out,
                                int* __restrict__ slot_node,
                                const NjBest* __restrict__ part,
                                int m, int n, int round,
                                int* __restrict__ merges)
{
    __shared__ NjBest sh[NJ_T_MAX];
    NjBest best{0.0, -1, -1};
    for (int a = (int)threadIdx.x; a < m - 1; a += (int)blockDim.x)
        if (nj_better(part[a], best)) best = part[a];
    sh[threadIdx.x] = best;
    __syncthreads();
    nj_block_reduce(sh);
    const int bi = sh[0].a, bj = sh[0].b;
    const int si = alive_in[bi], sj = alive_in[bj];
    const double* row_i = d + (size_t)si * (size_t)n;
    const double* row_j = d + (size_t)sj * (size_t)n;
    const double dij = row_i[sj];
    if (threadIdx.x == 0) {
        merges[2 * round]     = slot_node[si];
        merges[2 * round + 1] = slot_node[sj];
    }
    for (int k = (int)threadIdx.x; k < m; k += (int)blockDim.x) {
        if (k == bi || k == bj) continue;
        const int v = alive_in[k];
        const double nv = 0.5 * (row_i[v] + row_j[v] - dij);
        d[(size_t)si * (size_t)n + (size_t)v] = nv;
        d[(size_t)v * (size_t)n + (size_t)si] = nv;
        alive_out[k < bi ? k : (k < bj ? k - 1 : k - 2)] = v;
    }
    if (threadIdx.x == 0) {
        alive_out[m - 2] = si;
        slot_node[si]    = n + round;
    }
}

} // namespace genoaligner

#endif // GENOALIGNER_NJ_KERNEL_IMPL_HIP
