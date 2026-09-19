// NJ0 parity: nj_tree_gpu (the SHIPPED driver + kernel bodies) vs nj_tree.
// Tree.nodes must be identical element for element -- same children, same
// order, same root -- including on matrices built to have exact Q ties.
//
// CPU shim (no GPU):
//   g++ -O2 -std=c++17 -DGENOALIGNER_HIP_SHIM -I tests/parity/hip_cpu_shim
//       -I include -o /tmp/nj_gpu_test tests/msa/test_nj_gpu.cpp
//       src/msa/nj_gpu.cpp src/msa/msa_ref.cpp -lpthread
// Device:
//   hipcc -O2 -std=c++17 -Iinclude -o nj_gpu_test tests/msa/test_nj_gpu.cpp
//       src/msa/nj_gpu.cpp src/msa/msa_ref.cpp
// Exit 77 when no device is present (ctest SKIP_RETURN_CODE).
#include <genoaligner/msa/msa.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#ifdef GENOALIGNER_HIP_SHIM
thread_local uint3 threadIdx{0, 0, 0};
uint3 blockIdx{0, 0, 0};
dim3  blockDim{1, 1, 1};
dim3  gridDim{1, 1, 1};
#endif

using namespace genomsa;

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); ++fails; } } while (0)

static bool same_tree(const Tree& a, const Tree& b) {
    if (a.root != b.root || a.nodes.size() != b.nodes.size()) return false;
    for (size_t i = 0; i < a.nodes.size(); ++i)
        if (a.nodes[i].left != b.nodes[i].left || a.nodes[i].right != b.nodes[i].right)
            return false;
    return true;
}

static void check(const std::vector<float>& D, int n, const char* tag) {
    Tree ref = nj_tree(D, n);
    std::string err;
    NjStats st;
    Tree got = nj_tree_gpu(D, n, err, &st);
    CHECK(err.empty(), "%s: nj_tree_gpu error: %s", tag, err.c_str());
    CHECK(same_tree(ref, got), "%s: tree mismatch (n=%d)", tag, n);
    CHECK(n < 3 || st.rounds == n - 2, "%s: rounds=%d", tag, st.rounds);
    if (same_tree(ref, got) && err.empty()) printf("ok   %-28s n=%d\n", tag, n);
}

static std::mt19937 rng(20260913);

static std::vector<float> random_matrix(int n, float lo, float hi) {
    std::uniform_real_distribution<float> U(lo, hi);
    std::vector<float> D((size_t)n * (n - 1) / 2);
    for (auto& v : D) v = U(rng);
    return D;
}

// Distances on a small grid of values: many exact Q ties, so the first
// (a,b)-in-scan-order rule is exercised on nearly every round.
static std::vector<float> quantized_matrix(int n, int levels) {
    std::vector<float> D((size_t)n * (n - 1) / 2);
    for (auto& v : D) v = 0.5f + 0.5f * (float)(rng() % levels) / (float)levels;
    return D;
}

// All-equal distances: every pair ties in every round.
static std::vector<float> constant_matrix(int n, float c) {
    return std::vector<float>((size_t)n * (n - 1) / 2, c);
}

// Additive tree metric: NJ must recover it; exercises the "true" NJ path.
static std::vector<float> additive_matrix(int n) {
    std::vector<int> parent(2 * n - 1, -1);
    std::vector<double> blen(2 * n - 1, 0);
    std::vector<int> alive(n);
    for (int i = 0; i < n; ++i) alive[i] = i;
    int next = n;
    std::uniform_real_distribution<double> U(0.01, 0.2);
    while (alive.size() > 1) {
        int i = (int)(rng() % alive.size()), j;
        do { j = (int)(rng() % alive.size()); } while (j == i);
        parent[alive[i]] = next; parent[alive[j]] = next;
        blen[alive[i]] = U(rng); blen[alive[j]] = U(rng);
        if (i > j) std::swap(i, j);
        alive.erase(alive.begin() + j); alive.erase(alive.begin() + i);
        alive.push_back(next++);
    }
    auto path = [&](int x) {
        std::vector<int> p; while (x >= 0) { p.push_back(x); x = parent[x]; } return p;
    };
    std::vector<float> D((size_t)n * (n - 1) / 2);
    for (int a = 0; a < n; ++a) {
        auto pa = path(a);
        for (int b = a + 1; b < n; ++b) {
            auto pb = path(b);
            double s = 0;
            for (int x : pa) {
                bool shared = false;
                for (int y : pb) if (x == y) { shared = true; break; }
                if (shared) break;
                s += blen[x];
            }
            for (int y : pb) {
                bool shared = false;
                for (int x : pa) if (x == y) { shared = true; break; }
                if (shared) break;
                s += blen[y];
            }
            D[(size_t)a * n - (size_t)a * (a + 1) / 2 + (b - a - 1)] = (float)s;
        }
    }
    return D;
}

int main() {
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
        printf("SKIP: no HIP device\n");
        return 77;
    }

    // degenerate sizes
    check({}, 1, "n=1");
    check({0.3f}, 2, "n=2");
    check(random_matrix(3, 0.1f, 1.0f), 3, "n=3 random");
    check(random_matrix(4, 0.1f, 1.0f), 4, "n=4 random");

    for (int n : {5, 8, 17, 33, 64}) {
        check(random_matrix(n, 0.6f, 1.0f), n, "random kmer-like [0.6,1]");
        check(random_matrix(n, 0.0f, 1.0f), n, "random [0,1]");
        check(quantized_matrix(n, 4), n, "quantized 4 levels (ties)");
        check(quantized_matrix(n, 2), n, "quantized 2 levels (ties)");
        check(constant_matrix(n, 0.75f), n, "constant (all tie)");
        check(constant_matrix(n, 0.0f), n, "zeros (all tie)");
        check(additive_matrix(n), n, "additive tree metric");
    }
    // multi-block rowsum (> NJ_T rows). The shim spawns a host thread per
    // lane per block, so it runs a smaller instance.
#ifdef GENOALIGNER_HIP_SHIM
    const int big = 120;
#else
    const int big = 1000;
#endif
    check(random_matrix(big, 0.6f, 1.0f), big, "random big");
    check(quantized_matrix(big, 3), big, "quantized big (ties)");

    // error path: bad packed size must fail loudly, not guess
    {
        std::string err;
        Tree t = nj_tree_gpu(std::vector<float>(3, 0.5f), 5, err);
        CHECK(!err.empty() && t.nodes.empty(), "size mismatch not reported");
    }

    if (fails == 0) printf("ALL OK\n");
    else printf("%d FAILURES\n", fails);
    return fails;
}
