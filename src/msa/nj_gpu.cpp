// NJ0: exact dense Neighbor Joining on the device (docs/SPEC_NJ_GPU.md).
// nj_tree_gpu returns the SAME Tree as genomsa::nj_tree (bit-exact, same
// tie-break), or an empty Tree with `err` set -- never an approximate tree.
//
// Per round (m live nodes): rowsum (ceil(m/NJ_RS_T) blocks), argmin (m blocks),
// merge (1 block). No host<->device traffic inside the loop: the merges are
// recorded on the device and downloaded once, then the Tree is assembled on
// the host in O(n).
//
// Needs a HIP compiler (hipcc / nvcc -x cu), or the CPU shim
// (-DGENOALIGNER_HIP_SHIM -I tests/parity/hip_cpu_shim) which runs the same
// kernel bodies on host threads -- that is how tests/msa/test_nj_gpu.cpp
// gates parity without a GPU.

#include <genoaligner/msa/msa.hpp>
#include <genoaligner/backend/nj_kernel_impl.hip>
#include <hip/hip_runtime.h>
#include <chrono>
#include <string>
#include <vector>

namespace genomsa {

using genoaligner::NJ_T;
using genoaligner::NJ_RS_T;
using genoaligner::NjBest;
using genoaligner::nj_argmin_kernel;
using genoaligner::nj_merge_kernel;
using genoaligner::nj_rowsum_kernel;

namespace {

struct DevMem {
    std::vector<void*> ps;
    ~DevMem() { for (void* p : ps) if (p) (void)hipFree(p); }
    bool alloc(void** p, size_t bytes) {
        if (bytes == 0) bytes = 8;
        if (hipMalloc(p, bytes) != hipSuccess) { *p = nullptr; return false; }
        ps.push_back(*p);
        return true;
    }
};

#ifdef GENOALIGNER_HIP_SHIM
// Emulated launches: the shim has no scheduler, so the driver walks the grid;
// every block goes through shim::run_block (real threads, real barrier).
void launch_rowsum(unsigned grid, const double* d, const int* alive, int m, int n,
                   double* r) {
    gridDim = dim3{grid, 1, 1}; blockDim = dim3{(unsigned)NJ_RS_T, 1, 1};
    for (unsigned bx = 0; bx < grid; ++bx) {
        blockIdx = uint3{bx, 0, 0};
        shim::run_block(NJ_RS_T, [&] { nj_rowsum_kernel(d, alive, m, n, r); });
    }
}
void launch_argmin(unsigned grid, const double* d, const int* alive, const double* r,
                   int m, int n, NjBest* part) {
    gridDim = dim3{grid, 1, 1}; blockDim = dim3{(unsigned)NJ_T, 1, 1};
    for (unsigned bx = 0; bx < grid; ++bx) {
        blockIdx = uint3{bx, 0, 0};
        shim::run_block(NJ_T, [&] { nj_argmin_kernel(d, alive, r, m, n, part); });
    }
}
void launch_merge(double* d, const int* ain, int* aout, int* slot_node,
                  const NjBest* part, int m, int n, int round, int* merges) {
    gridDim = dim3{1, 1, 1}; blockDim = dim3{(unsigned)NJ_T, 1, 1};
    blockIdx = uint3{0, 0, 0};
    shim::run_block(NJ_T, [&] {
        nj_merge_kernel(d, ain, aout, slot_node, part, m, n, round, merges);
    });
}
#else
void launch_rowsum(unsigned grid, const double* d, const int* alive, int m, int n,
                   double* r) {
    hipLaunchKernelGGL(nj_rowsum_kernel, dim3(grid), dim3(NJ_RS_T), 0, nullptr,
                       d, alive, m, n, r);
}
void launch_argmin(unsigned grid, const double* d, const int* alive, const double* r,
                   int m, int n, NjBest* part) {
    hipLaunchKernelGGL(nj_argmin_kernel, dim3(grid), dim3(NJ_T), 0, nullptr,
                       d, alive, r, m, n, part);
}
void launch_merge(double* d, const int* ain, int* aout, int* slot_node,
                  const NjBest* part, int m, int n, int round, int* merges) {
    hipLaunchKernelGGL(nj_merge_kernel, dim3(1), dim3(NJ_T), 0, nullptr,
                       d, ain, aout, slot_node, part, m, n, round, merges);
}
#endif

} // namespace

Tree nj_tree_gpu(const std::vector<float>& Dp, int n, std::string& err,
                 NjStats* stats) {
    using clock = std::chrono::steady_clock;
    auto secs = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    err.clear();
    Tree t;
    if (n <= 0) { err = "nj_tree_gpu: n <= 0"; return Tree{}; }
    if (n == 1) { t.nodes.resize(1); t.root = 0; return t; }
    if (n == 2) {
        t.nodes.resize(2); t.root = 2; t.nodes.push_back({0, 1});
        return t;
    }
    const size_t npairs = (size_t)n * (n - 1) / 2;
    if (Dp.size() != npairs) { err = "nj_tree_gpu: packed size mismatch"; return Tree{}; }

    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
        err = "nj_tree_gpu: no HIP device";
        return Tree{};
    }

    auto t0 = clock::now();
    // Dense symmetric double matrix, same conversion as nj_tree (float -> double).
    std::vector<double> hd((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        const size_t base = (size_t)i * n - (size_t)i * (i + 1) / 2;
        for (int j = i + 1; j < n; ++j) {
            const double v = Dp[base + (size_t)(j - i - 1)];
            hd[(size_t)i * n + j] = v;
            hd[(size_t)j * n + i] = v;
        }
    }
    std::vector<int> hid(n);
    for (int i = 0; i < n; ++i) hid[i] = i;

    DevMem mem;
    double* d_d = nullptr; double* d_r = nullptr;
    int* d_alive[2] = {nullptr, nullptr};
    int* d_slot = nullptr; int* d_merges = nullptr;
    NjBest* d_part = nullptr;
    const size_t bytes_d = hd.size() * sizeof(double);
    const size_t bytes_i = (size_t)n * sizeof(int);
    size_t total = 0;
    auto mal = [&](void** p, size_t b, const char* what) {
        if (!mem.alloc(p, b)) { err = std::string("nj_tree_gpu: hipMalloc ") + what; return false; }
        total += b;
        return true;
    };
    if (!mal((void**)&d_d, bytes_d, "dist") ||
        !mal((void**)&d_r, (size_t)n * sizeof(double), "rowsum") ||
        !mal((void**)&d_alive[0], bytes_i, "alive0") ||
        !mal((void**)&d_alive[1], bytes_i, "alive1") ||
        !mal((void**)&d_slot, bytes_i, "slot_node") ||
        !mal((void**)&d_merges, 2 * bytes_i, "merges") ||
        !mal((void**)&d_part, (size_t)n * sizeof(NjBest), "partials"))
        return Tree{};
    if (stats) stats->dev_bytes = total;

    if (hipMemcpy(d_d, hd.data(), bytes_d, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d_alive[0], hid.data(), bytes_i, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d_slot, hid.data(), bytes_i, hipMemcpyHostToDevice) != hipSuccess) {
        err = "nj_tree_gpu: hipMemcpy H2D";
        return Tree{};
    }
    auto t1 = clock::now();

    int cur = 0;
    const int rounds = n - 2;
    for (int round = 0; round < rounds; ++round) {
        const int m = n - round;
        launch_rowsum((unsigned)((m + NJ_RS_T - 1) / NJ_RS_T), d_d, d_alive[cur], m, n, d_r);
        launch_argmin((unsigned)m, d_d, d_alive[cur], d_r, m, n, d_part);
        launch_merge(d_d, d_alive[cur], d_alive[cur ^ 1], d_slot, d_part, m, n,
                     round, d_merges);
        cur ^= 1;
    }
    {
        hipError_t le = hipGetLastError();
        hipError_t se = hipDeviceSynchronize();
        if (le != hipSuccess || se != hipSuccess) {
            err = std::string("nj_tree_gpu: kernel: ") +
                  hipGetErrorString(le != hipSuccess ? le : se);
            return Tree{};
        }
    }
    auto t2 = clock::now();

    std::vector<int> merges((size_t)2 * rounds), slot(n), alive2(2);
    if (hipMemcpy(merges.data(), d_merges, merges.size() * sizeof(int),
                  hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(slot.data(), d_slot, bytes_i, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(alive2.data(), d_alive[cur], 2 * sizeof(int),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        err = "nj_tree_gpu: hipMemcpy D2H";
        return Tree{};
    }
    auto t3 = clock::now();

    t.nodes.resize(n);
    t.nodes.reserve((size_t)2 * n - 1);
    for (int round = 0; round < rounds; ++round) {
        const int xi = merges[2 * round], xj = merges[2 * round + 1];
        if (xi < 0 || xj < 0 || xi >= n + round || xj >= n + round || xi == xj) {
            err = "nj_tree_gpu: corrupt merge record";
            return Tree{};
        }
        t.nodes.push_back({xi, xj});
    }
    t.root = (int)t.nodes.size();
    t.nodes.push_back({slot[alive2[0]], slot[alive2[1]]});

    if (stats) {
        stats->upload_s   = secs(t0, t1);
        stats->rounds_s   = secs(t1, t2);
        stats->download_s = secs(t2, t3);
        stats->rounds     = rounds;
        stats->exact_fallbacks = 0;
    }
    return t;
}

} // namespace genomsa
