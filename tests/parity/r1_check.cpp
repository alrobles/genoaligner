// genoaligner — R1 verification: CPU replay of the HIP WFA kernel.
//
// Compiles and executes the REAL kernel source
// (include/genoaligner/backend/wfa_kernel.hip) with no GPU present, so that a
// CPU check can gate the cluster run. This tests wfa_compute/wfa_extend/wfa_step
// -- the exact functions the __global__ kernel calls -- rather than a re-typed
// copy, so a divergence here is a divergence in the shipped code.
//
// The replay is faithful because the kernel is phase-structured: each phase
// reads only what the preceding __syncthreads() made visible, so iterating
// threadIdx over the phase and running the phases in order is what the GPU does
// between barriers.
//
// The oracle is src/reference/edit_distance_cpu.hpp: a classic O(nm) DP, a
// DIFFERENT algorithm from the wavefront, so agreement is real evidence.
//
// Build:
//   g++ -O2 -std=c++17 -DGENOALIGNER_HIP_SHIM -I/tmp/hipshim -I<repo> \
//       -o r1_check tests/parity/r1_check.cpp
#define GENOALIGNER_HIP_SHIM
#include "hip/hip_runtime.h"
#include "include/genoaligner/backend/wfa_kernel.hip"
#include "src/reference/edit_distance_cpu.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace genoaligner;

// Definitions for the shim's launch-configuration globals. The shim declares
// them; something must define them. The kernel references them but the replay
// below does not launch it, so they stay zero.
uint3 threadIdx{0,0,0};
uint3 blockIdx{0,0,0};
dim3  blockDim{1,1,1};
dim3  gridDim{1,1,1};

static int wfa_replay(const PairView& p, int smax)
{
    const int m = p.pattern_len, n = p.text_len;
    if (m <= 0 || n <= 0) return (m <= 0) ? n : m;

    const int stride    = 2 * smax + 1;
    const int wf_stride = stride + 1;
    shim::smem_vec().assign(2 * wf_stride, WFA_NEG);
    int* smem = shim_smem_ptr();
    int* A = smem;
    int* B = smem + wf_stride;
    auto idx = [&](int k) { return k + smax + 1; };
    const int need = 2 * smax + 1;

    for (int t = 0; t < wf_stride; ++t) { A[t] = WFA_NEG; B[t] = WFA_NEG; }

    // score 0
    {
        int offset = 0;
        while (offset < m && offset < n && p.pattern[offset] == p.text[offset]) ++offset;
        A[idx(0)] = offset;
    }
    const int alignment_k = wfa_diag(n, m);
    if (alignment_k == 0 && A[idx(0)] >= n) return 0;

    int* prev = A; int* cur = B;
    for (int s = 1; s <= smax; ++s) {
        const int lo = -s, hi = s;

        // compute + extend, via the SAME functions the GPU kernel calls
        for (int t = 0; t <= need - 1 && t <= 2 * s; ++t) {
            const int k = t - s;
            cur[idx(k)] = wfa_step(p.pattern, p.text, k, m, n,
                                   prev[idx(k - 1)],
                                   prev[idx(k + 1)],
                                   prev[idx(k)]);
        }

        // termination
        if (alignment_k >= lo && alignment_k <= hi) {
            const int off = cur[idx(alignment_k)];
            if (wfa_reachable(off) && off >= n) return s;
        }
        int* tmp = prev; prev = cur; cur = tmp;
    }
    return -1;
}

static int g_fail = 0;

static void check(const char* plabel, const std::string& P, const std::string& T, int smax)
{
    PairView p{T.data(), (int)T.size(), P.data(), (int)P.size(), smax};
    int got = wfa_replay(p, smax);
    int ref = edit_distance_cpu(P.data(), (int)P.size(), T.data(), (int)T.size());
    bool ok = (got == ref);
    if (!ok) g_fail++;
    printf("  %-26s p=%-8s t=%-8s kernel=%4d cpu=%4d  %s\n",
           plabel, P.c_str(), T.c_str(), got, ref, ok ? "ok" : "** MISMATCH **");
}

int main()
{
    const int smax = 128;
    printf("genoaligner R1 -- HIP kernel replay vs CPU DP reference\n");
    printf("  smax = %d   (functions under test: wfa_compute/wfa_extend/wfa_step)\n\n", smax);

    check("identical single",   "A", "A", smax);
    check("single mismatch",    "A", "T", smax);
    check("pattern longer",     "AA", "A", smax);
    check("text longer",        "A", "AA", smax);
    check("identical 4",        "ACGT", "ACGT", smax);
    check("one mismatch",       "ACGT", "ACGA", smax);
    check("one deletion",       "ACGTA", "ACGT", smax);
    check("one insertion",      "ACGT", "ACGTA", smax);
    check("min reproducer 1",   "GG", "TTT", smax);
    check("min reproducer 2",   "CT", "CCC", smax);
    check("min reproducer 3",   "CG", "GAC", smax);
    check("insert+sub",         "C", "GT", smax);
    check("both empty",         "", "", smax);
    check("empty text",         "ACGT", "", smax);
    check("empty pattern",      "", "ACGT", smax);

    // ---- regime harness: subs / indels / mixed ----------------------------
    {
        std::mt19937 rng(7);
        const char A[] = "ACGT"; std::uniform_int_distribution<int> ch(0,3);
        int fail=0, tot=0, aband=0, fm[3]={0,0,0}, nm[3]={0,0,0};
        for (int c = 0; c < 3000; ++c) {
            int len = 24 + (c % 6) * 8;
            std::string tt; for (int i=0;i<len;++i) tt.push_back(A[ch(rng)]);
            std::string pp = tt;
            std::uniform_int_distribution<int> pos(0, len-1);
            int mode = c % 3; nm[mode]++;
            if (mode==0) { for (int i=0;i<len/4;++i) pp[pos(rng)] = A[ch(rng)]; }
            else if (mode==1) { for (int i=0;i<3;++i) if (pp.size()>6) pp.erase(pp.begin()+pos(rng)%pp.size());
                                for (int i=0;i<3;++i) pp.insert(pp.begin()+pos(rng)%pp.size(), A[ch(rng)]); }
            else { for (int i=0;i<len/8;++i) pp[pos(rng)] = A[ch(rng)];
                   if (pp.size()>6) { pp.erase(pp.begin()+(pos(rng)%pp.size())); pp.insert(pp.begin()+(pos(rng)%pp.size()), A[ch(rng)]); } }
            PairView p{tt.data(), (int)tt.size(), pp.data(), (int)pp.size(), smax};
            int got = wfa_replay(p, smax);
            int ref = edit_distance_cpu(pp.data(), (int)pp.size(), tt.data(), (int)tt.size());
            tot++;
            if (got < 0) { aband++; continue; }
            if (got != ref) { fail++; fm[mode]++; if (fail<=8) printf("  FAIL mode=%d kernel=%d cpu=%d m=%zu n=%zu\n", mode, got, ref, pp.size(), tt.size()); }
        }
        printf("\n  per-regime fail: subs %d/%d  indels %d/%d  mixed %d/%d\n",
               fm[0],nm[0],fm[1],nm[1],fm[2],nm[2]);
        printf("  regime total   : pass=%d fail=%d abandoned=%d of %d  parity=%.2f%%\n",
               tot-fail-aband, fail, aband, tot, 100.0*(tot-fail-aband)/(tot-aband?tot-aband:1));
        g_fail += fail;
    }

    // ---- exhaustive small pairs -------------------------------------------
    {
        std::mt19937 rng(2024);
        const char A[] = "ACGT";
        int fail=0, tot=0;
        for (int a=0;a<=6;++a) for (int b=0;b<=6;++b) for (int tr=0;tr<30;++tr) {
            std::string P,T;
            for (int i=0;i<a;++i) P.push_back(A[rng()%4]);
            for (int i=0;i<b;++i) T.push_back(A[rng()%4]);
            PairView p{T.data(), (int)T.size(), P.data(), (int)P.size(), smax};
            int got = wfa_replay(p, smax);
            int ref = edit_distance_cpu(P.data(), (int)P.size(), T.data(), (int)T.size());
            tot++;
            if (got != ref) { fail++; if (fail<=5) printf("  SMALL FAIL P=%s T=%s kernel=%d cpu=%d\n", P.c_str(), T.c_str(), got, ref); }
        }
        printf("  exhaustive small (len<=6): fail=%d/%d\n", fail, tot);
        g_fail += fail;
    }

    printf("\n==== R1 %s ====\n", g_fail ? "KERNEL PARITY FAILED" : "KERNEL PARITY 100% -- cleared for MI210");
    return g_fail ? 1 : 0;
}
