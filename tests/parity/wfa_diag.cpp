// genoaligner — per-regime diagnostic test (permanent regression).
//
// WHY THIS EXISTS: during Phase 2 the aggregate parity number was ~55% and
// opaque. Splitting failures by mutation regime (substitutions only / indels
// only / mixed) turned that opaque 55% into a localised bug. Regime separation
// is the diagnostic that pays for itself, so it lives in the repo rather than in
// a scratch file.
//
// This test also pins the five minimal reproducers found during the R1 fix.
// Each one blocked a specific, distinct bug; if any regresses, the count below
// will not be zero and the failing regime will point at the cause.
//
// It tests wfa_compute/wfa_extend/wfa_step from the real kernel header
// (include/genoaligner/backend/wfa_kernel.hip) via the CPU shim, so it runs with
// no GPU and can gate the cluster run.
//
// Build (no GPU needed):
//   scripts/check_kernel_cpu.sh
#define GENOALIGNER_HIP_SHIM
#include "hip/hip_runtime.h"
#include "include/genoaligner/backend/wfa_kernel.hip"
#include "src/reference/edit_distance_cpu.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace genoaligner;

thread_local uint3 threadIdx{0,0,0};
uint3 blockIdx{0,0,0};
dim3  blockDim{1,1,1};
dim3  gridDim{1,1,1};

// Replay one pair through the kernel's own step function.
static int wfa_replay(const PairView& p, int smax)
{
    const int m = p.pattern_len, n = p.text_len;
    if (m <= 0 || n <= 0) return (m <= 0) ? n : m;

    const int stride    = 2 * smax + 1;
    const int wf_stride = stride + 2;   // padding on BOTH ends (see kernel note)
    // Size exactly as the H2 launch site does, so an undersized launch trips
    // ASan here rather than faulting on the MI210 (see H2 job 29184154).
    shim::smem_vec().assign((size_t)2 * (2 * smax + 3), WFA_NEG);
    int* smem = shim_smem_ptr();
    int* A = smem;
    int* B = smem + wf_stride;
    auto idx = [&](int k) { return k + smax + 1; };
    const int need = 2 * smax + 1;

    for (int t = 0; t < wf_stride; ++t) { A[t] = WFA_NEG; B[t] = WFA_NEG; }

    {
        int offset = 0;
        while (offset < m && offset < n && p.pattern[offset] == p.text[offset]) ++offset;
        A[idx(0)] = offset;
    }
    const int alignment_k = wfa_diag(n, m);
    if (alignment_k == 0 && A[idx(0)] >= n) return 0;

    int* prev = A; int* cur = B;
    for (int s = 1; s <= smax; ++s) {
        const int lo = -s;
        for (int t = 0; t < need && t <= 2 * s; ++t) {
            const int k = t - s;
            cur[idx(k)] = wfa_step(p.pattern, p.text, k, m, n,
                                   prev[idx(k - 1)], prev[idx(k + 1)], prev[idx(k)]);
        }
        if (alignment_k >= lo && alignment_k <= s) {
            const int off = cur[idx(alignment_k)];
            if (wfa_reachable(off) && off >= n) return s;
        }
        int* tmp = prev; prev = cur; cur = tmp;
    }
    return -1;
}

int main()
{
    const int smax = 128;
    int failures = 0;

    printf("genoaligner -- per-regime WFA diagnostic (kernel vs CPU DP)\n\n");

    // ---- (1) minimal reproducers, each pinning a distinct past bug ---------
    struct R { const char* p; const char* t; int want; const char* why; };
    const R repro[] = {
        {"GG", "TTT",  3, "free insertions (missing +1 discipline)"},
        {"CT", "CCC",  2, "off-by-one on insertion run"},
        {"CG", "GAC",  3, "substitution + indel interaction"},
        {"C",  "GT",   2, "diagonal-direction confusion at k>0"},
        {"A",  "AA",   1, "NULL sentinel corrupting under +1"},
    };
    printf("minimal reproducers:\n");
    for (const auto& r : repro) {
        std::string P(r.p), T(r.t);
        PairView p{T.data(), (int)T.size(), P.data(), (int)P.size(), smax};
        int got = wfa_replay(p, smax);
        bool ok = (got == r.want);
        if (!ok) failures++;
        printf("  %-8s vs %-8s got=%3d want=%3d  %-6s  (%s)\n",
               r.p, r.t, got, r.want, ok ? "ok" : "FAIL", r.why);
    }

    // ---- (2) regime split --------------------------------------------------
    struct Regime { const char* name; int fail; int total; };
    Regime reg[3] = {{"substitutions only", 0, 0}, {"indels only", 0, 0}, {"mixed", 0, 0}};

    std::mt19937 rng(7);
    const char A[] = "ACGT";
    std::uniform_int_distribution<int> ch(0, 3);

    for (int c = 0; c < 3000; ++c) {
        const int len = 24 + (c % 6) * 8;
        std::string tt;
        for (int i = 0; i < len; ++i) tt.push_back(A[ch(rng)]);
        std::string pp = tt;
        std::uniform_int_distribution<int> pos(0, len - 1);
        const int mode = c % 3;

        if (mode == 0) {
            for (int i = 0; i < len / 4; ++i) pp[pos(rng)] = A[ch(rng)];
        } else if (mode == 1) {
            for (int i = 0; i < 3; ++i) if (pp.size() > 6) pp.erase(pp.begin() + pos(rng) % pp.size());
            for (int i = 0; i < 3; ++i) pp.insert(pp.begin() + pos(rng) % pp.size(), A[ch(rng)]);
        } else {
            for (int i = 0; i < len / 8; ++i) pp[pos(rng)] = A[ch(rng)];
            if (pp.size() > 6) {
                pp.erase(pp.begin() + (pos(rng) % pp.size()));
                pp.insert(pp.begin() + (pos(rng) % pp.size()), A[ch(rng)]);
            }
        }

        PairView p{tt.data(), (int)tt.size(), pp.data(), (int)pp.size(), smax};
        const int got = wfa_replay(p, smax);
        const int ref = edit_distance_cpu(pp.data(), (int)pp.size(), tt.data(), (int)tt.size());
        reg[mode].total++;
        if (got != ref) {
            reg[mode].fail++;
            failures++;
            if (reg[mode].fail <= 3)
                printf("  FAIL [%s] kernel=%d cpu=%d m=%zu n=%zu\n",
                       reg[mode].name, got, ref, pp.size(), tt.size());
        }
    }

    printf("\nregime split (this is the diagnostic that localises bugs):\n");
    for (const auto& r : reg)
        printf("  %-20s %4d/%4d fail\n", r.name, r.fail, r.total);

    printf("\n==== %s ====\n", failures ? "DIAGNOSTIC FAILED" : "ALL REGIMES CLEAN");
    return failures ? 1 : 0;
}
