// Fase 7 — how many wavefronts does the SCORE kernel actually walk?
//
// WHY THIS EXISTS
// ---------------
// Attribution (job 29210725, MI210) showed kernel time scaling ~linearly with
// smax at CONSTANT real distance:
//
//   len=1024, ident=98%, mean distance 15.2, smax 32->256
//     0.966 ms -> 2.190 -> 4.835 -> 10.349 ms      (walked 47% -> 24% -> 12% -> 6%)
//
// The kernel is supposed to RETURN as soon as a pair resolves (off >= n on the
// diagonal), so at distance 15 the time should be flat in smax. It is not. Either
// the early return never fires, or each wavefront costs more as blockDim grows.
// Those imply completely different fixes, so it gets measured rather than guessed.
//
// METHOD
// ------
// This replays the kernel's exact phase structure (the same shape r1_check.cpp
// uses and the same wfa_step the kernel calls) and counts the wavefronts actually
// walked per pair, comparing that to the pair's true distance. No GPU; runs in
// seconds. If exit_s == distance, the early return works and the cost is the
// per-wavefront block overhead. If exit_s ~= smax, the bar is unmet wavefronts.
//
// Build (no GPU, no ROCm):
//   g++ -O2 -std=c++17 -DGENOALIGNER_HIP_SHIM -Itests/parity/hip_cpu_shim -I. \
//       -o /tmp/h7exit bench/h7_exit_probe.cpp

#include <hip/hip_runtime.h>

#include "include/genoaligner/backend/wfa_kernel.hip"
#include "src/reference/edit_distance_cpu.hpp"

uint3 threadIdx{0, 0, 0};
uint3 blockIdx{0, 0, 0};
dim3  blockDim{1, 1, 1};
dim3  gridDim{1, 1, 1};

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>

using namespace genoaligner;

// Replay of wfa_score_kernel's phases, returning (score, wavefronts_walked).
// Mirrors tests/parity/r1_check.cpp::wfa_replay, which is the already-validated
// transcription of this kernel's algebra, plus a counter.
static void replay(const PairView& p, int smax, int* out_score, int* out_wf)
{
    const int m = p.pattern_len, n = p.text_len;
    if (m <= 0 || n <= 0) { *out_score = (m <= 0) ? n : m; *out_wf = 0; return; }

    const int stride    = 2 * smax + 1;
    const int wf_stride = stride + 2;
    const size_t launch_ints = (size_t)2 * (2 * smax + 3);
    shim::smem_vec().assign(launch_ints, WFA_NEG);
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
    if (alignment_k == 0 && A[idx(0)] >= n) { *out_score = 0; *out_wf = 0; return; }

    int* prev = A; int* cur = B;
    for (int s = 1; s <= smax; ++s) {
        const int lo = -s, hi = s;
        for (int t = 0; t <= need - 1 && t <= 2 * s; ++t) {
            const int k = t - s;
            cur[idx(k)] = wfa_step(p.pattern, p.text, k, m, n,
                                   prev[idx(k - 1)], prev[idx(k + 1)], prev[idx(k)]);
        }
        if (alignment_k >= lo && alignment_k <= hi) {
            const int off = cur[idx(alignment_k)];
            if (wfa_reachable(off) && off >= n) { *out_score = s; *out_wf = s; return; }
        }
        int* tmp = prev; prev = cur; cur = tmp;
    }
    *out_score = -1; *out_wf = smax;   // swept the whole range without resolving
}

static std::vector<std::string> gen(int np, int len, int ident, uint32_t seed,
                                    std::vector<std::string>* pats)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> base(0, 3), pos(0, len - 1), ch(0, 3);
    const char alpha[] = "ACGT";
    std::vector<std::string> texts(np);
    pats->resize(np);
    for (int c = 0; c < np; ++c) {
        std::string t;
        for (int i = 0; i < len; ++i) t.push_back(alpha[base(rng)]);
        std::string p = t;
        const int nmut = (int)((size_t)len * (100 - ident) / 100.0);
        for (int i = 0; i < nmut; ++i) p[pos(rng)] = alpha[ch(rng)];
        texts[c] = std::move(t);
        (*pats)[c] = std::move(p);
    }
    return texts;
}

int main()
{
    const int len = 1024, ident = 98, npairs = 400;
    std::vector<std::string> pats;
    std::vector<std::string> texts = gen(npairs, len, ident, 12345u, &pats);

    std::vector<PairView> views(npairs);
    for (int i = 0; i < npairs; ++i) {
        pats[i] += '\0'; texts[i] += '\0';   // stable storage, no reallocation
        views[i] = PairView{texts[i].data(), (int)texts[i].size() - 1,
                            pats[i].data(),  (int)pats[i].size() - 1, 0};
    }

    printf("len=%d ident=%d%% pairs=%d\n", len, ident, npairs);
    printf("smax sweep: how many wavefronts are actually walked\n\n");
    printf("%6s %10s %10s %10s %12s\n", "smax", "dist_mean", "wf_mean", "wf_max", "wf/dist");
    for (int smax : {32, 64, 128, 256}) {
        long dsum = 0, wsum = 0; int wmax = 0, res = 0, wrong = 0;
        for (int i = 0; i < npairs; ++i) {
            int sc = 0, wf = 0;
            replay(views[i], smax, &sc, &wf);
            if (sc < 0) { wsum += wf; continue; }
            const int want = edit_distance_cpu(pats[i].data(), (int)pats[i].size() - 1,
                                               texts[i].data(), (int)texts[i].size() - 1);
            if (sc != want) ++wrong;
            ++res; dsum += sc; wsum += wf;
            if (wf > wmax) wmax = wf;
        }
        const double dm = res ? (double)dsum / res : 0;
        const double wm = (double)wsum / npairs;
        printf("%6d %10.1f %10.1f %10d %12.2f   resolved=%d wrong=%d\n",
               smax, dm, wm, wmax, dm > 0 ? wm / dm : 0.0, res, wrong);
    }
    printf("\nIf wf_mean tracks dist_mean (ratio ~1) the EARLY RETURN IS FIRING and the\n"
           "kernel walks ~distance wavefronts regardless of smax. Time growth with smax\n"
           "would then be per-wavefront block overhead (blockDim = 2*smax+1), not empty work.\n");
    printf("If wf_mean ~= smax the loop is sweeping the full range: wasted wavefronts.\n");
    return 0;
}
