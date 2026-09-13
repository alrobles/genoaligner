// genoaligner — Smith-Waterman traceback on GPU (Fase B).
//
// What this run establishes, stated narrowly:
//   * sw_trace_kernel (one thread per pair, direction-byte table) produces on a
//     real device the same scores as sw_score_kernel (warp scan) and the host
//     reference, AND CIGARs that are well-formed on the aligned span, re-score
//     exactly, and equal the independent matrix walk;
//   * --emit writes the TSV the SeqAn3 oracle (tests/sw/seqan3_sw_oracle.cpp)
//     scores on the cluster -- the external leg of this gate.
//
// The CPU gate (tests/sw/test_sw_trace.cpp) already ran the same bodies through
// the shim; this is the check that nothing is lost on a real device, which is
// where every SW GPU fault so far has lived.

#include "genoaligner/backend/sw_kernel.hip"
#include "genoaligner/backend/sw_kernel_impl.hip"
#include "../parity/cigar.hpp"
#include "sw_cigar.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include <random>

using genoaligner::SWParams;
using genoaligner::SWPairView;
using genoaligner::SWResult;
using genoaligner::SWTraceResult;
using genoaligner::sw_score_kernel;
using genoaligner::sw_trace_kernel;

static int g_fail = 0;

// Same independent reference as the CPU gate: full matrices + a value walk.
struct RefOut {
    int score = 0;
    int start_i = -1, start_j = -1, end_i = -1, end_j = -1;
    std::vector<int> cigar_rev;
};

static RefOut ref_sw_trace(const std::string& text, const std::string& pattern,
                           SWParams p)
{
    RefOut r;
    const int m = (int)text.size(), n = (int)pattern.size();
    if (m == 0 || n == 0) return r;
    const int W = n + 1;
    std::vector<int> H((size_t)(m + 1) * W, 0), E(H.size(), 0), F(H.size(), 0);
    auto at = [](std::vector<int>& M, int i, int j, int w) -> int& {
        return M[(size_t)i * w + (size_t)j];
    };
    int best = 0, bi = 0, bj = 0;
    for (int i = 1; i <= m; ++i)
        for (int j = 1; j <= n; ++j) {
            const int s = (text[(size_t)i - 1] == pattern[(size_t)j - 1]) ? p.match : p.mismatch;
            int e = at(H, i, j - 1, W) - p.gap_open;
            { const int t = at(E, i, j - 1, W) - p.gap_extend; if (t > e) e = t; }
            int f = at(H, i - 1, j, W) - p.gap_open;
            { const int t = at(F, i - 1, j, W) - p.gap_extend; if (t > f) f = t; }
            int h = at(H, i - 1, j - 1, W) + s;
            if (e > h) h = e;
            if (f > h) h = f;
            if (h < 0) h = 0;
            at(H, i, j, W) = h; at(E, i, j, W) = e; at(F, i, j, W) = f;
            if (h > best) { best = h; bi = i; bj = j; }
        }
    r.score = best;
    if (best == 0) return r;
    int i = bi, j = bj, state = 0;
    while (i > 0 && j > 0) {
        if (state == 0) {
            const int h = at(H, i, j, W);
            if (h == 0) break;
            const int s = (text[(size_t)i - 1] == pattern[(size_t)j - 1]) ? p.match : p.mismatch;
            if (h == at(H, i - 1, j - 1, W) + s) {
                r.cigar_rev.push_back(text[(size_t)i - 1] == pattern[(size_t)j - 1]
                                      ? genoaligner::SW_OP_M : genoaligner::SW_OP_X);
                --i; --j;
            } else if (h == at(E, i, j, W)) {
                r.cigar_rev.push_back(genoaligner::SW_OP_D);
                const bool ext = (at(E, i, j - 1, W) - p.gap_extend)
                               > (at(H, i, j - 1, W) - p.gap_open);
                --j; state = ext ? 1 : 0;
            } else {
                r.cigar_rev.push_back(genoaligner::SW_OP_I);
                const bool ext = (at(F, i - 1, j, W) - p.gap_extend)
                               > (at(H, i - 1, j, W) - p.gap_open);
                --i; state = ext ? 2 : 0;
            }
        } else if (state == 1) {
            r.cigar_rev.push_back(genoaligner::SW_OP_D);
            const bool ext = (at(E, i, j - 1, W) - p.gap_extend)
                           > (at(H, i, j - 1, W) - p.gap_open);
            --j; if (!ext) state = 0;
        } else {
            r.cigar_rev.push_back(genoaligner::SW_OP_I);
            const bool ext = (at(F, i - 1, j, W) - p.gap_extend)
                           > (at(H, i - 1, j, W) - p.gap_open);
            --i; if (!ext) state = 0;
        }
    }
    r.start_i = i; r.start_j = j; r.end_i = bi - 1; r.end_j = bj - 1;
    return r;
}

struct Case { std::string text, pattern, label; };

int main(int argc, char** argv)
{
    const char* emit_path = nullptr;
    for (int a = 1; a < argc; ++a)
        if (std::strcmp(argv[a], "--emit") == 0 && a + 1 < argc) emit_path = argv[++a];

    printf("genoaligner — SW traceback on GPU (score + CIGAR)\n");

    int nsm = 0;
    hipDeviceProp_t prop{};
    if (hipGetDeviceCount(&nsm) != hipSuccess || nsm == 0 ||
        hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        printf("  SKIP: no device visible.\nRESULT: SKIP -- needs a GPU (exit 77)\n");
        return 77;
    }
    int dev_warp = 0;
    hipDeviceGetAttribute(&dev_warp, hipDeviceAttributeWarpSize, 0);
    printf("  device: %s   warpSize=%d\n\n", prop.name, dev_warp);

    const SWParams P{1, -1, 2, 1};

    std::vector<Case> cases = {
        {"A", "A", "single base"},
        {"ACGTACGT", "ACGTACGT", "8 identical"},
        {"ACGTACGT", "TGCATGCA", "shares one A"},
        {"AAAAAAAA", "CCCCCCCC", "disjoint"},
        {"ACGTTTACGT", "ACGTACGT", "gap of 2 in pattern (I run)"},
        {"ACGTACGT", "ACGTTTACGT", "gap of 2 in text (D run)"},
        {"AACGTTTACGTAA", "TTCGTACGTT", "gap inside shifted flanks"},
        {"ACACACAC", "ACAC", "tandem repeat"},
        {"", "", "both empty"},
        {"ACGT", "", "empty pattern"},
        {"", "ACGT", "empty text"},
        {std::string("ACGT") + std::string(6, 'T') + "ACGT",
         std::string("ACGT") + "ACGT", "long gap in pattern (6I)"},
        {std::string("ACGT") + "ACGT",
         std::string("ACGT") + std::string(6, 'T') + "ACGT", "long gap in text (6D)"},
        {"ACGTAAACGTGGGACGT", "ACGTACGTACGT", "two gaps"},
        {std::string(33,'A'), std::string(33,'A'), "n=33"},
        {std::string(65,'A'), std::string(65,'C'), "n=65 disjoint tail"},
    };
    {
        std::string t, q;
        for (int i = 0; i < 400; ++i) t.push_back("ACGT"[i % 4]);
        q = std::string(20,'T') + t.substr(100, 120) + std::string(20,'G');
        cases.push_back({t, q, "400 x 160 embedded core"});
    }
    {
        std::mt19937 rng(20260912u);
        const char* alpha = "ACGT";
        for (int k = 0; k < 192; ++k) {
            const int m = (int)(rng() % 201), n = (int)(rng() % 201);
            std::string t, q;
            for (int i = 0; i < m; ++i) t.push_back(alpha[rng() % 4]);
            for (int i = 0; i < n; ++i) q.push_back(alpha[rng() % 4]);
            cases.push_back({t, q, "random"});
        }
    }

    const int N = (int)cases.size();

    // Device buffers: sequences as device pointers (the SWPairView pointer bug
    // cost a job once; build views only from device memory).
    std::vector<SWPairView> hv((size_t)N);
    std::vector<char*> d_txt((size_t)N, nullptr), d_pat((size_t)N, nullptr);
    for (int i = 0; i < N; ++i) {
        const std::string& t = cases[(size_t)i].text;
        const std::string& q = cases[(size_t)i].pattern;
        if (hipMalloc(&d_txt[(size_t)i], t.size() ? t.size() : 1) != hipSuccess ||
            hipMalloc(&d_pat[(size_t)i], q.size() ? q.size() : 1) != hipSuccess) {
            printf("  hipMalloc(chars) failed\n"); return 1;
        }
        hipMemcpy(d_txt[(size_t)i], t.data(), t.size(), hipMemcpyHostToDevice);
        hipMemcpy(d_pat[(size_t)i], q.data(), q.size(), hipMemcpyHostToDevice);
        hv[(size_t)i].text = d_txt[(size_t)i]; hv[(size_t)i].text_len = (int)t.size();
        hv[(size_t)i].pattern = d_pat[(size_t)i]; hv[(size_t)i].pattern_len = (int)q.size();
    }

    SWPairView* d_pairs = nullptr;
    SWResult*   d_res   = nullptr;
    if (hipMalloc(&d_pairs, (size_t)N * sizeof(SWPairView)) != hipSuccess ||
        hipMalloc(&d_res,   (size_t)N * sizeof(SWResult))   != hipSuccess) {
        printf("  hipMalloc failed\n"); return 1;
    }
    hipMemcpy(d_pairs, hv.data(), (size_t)N * sizeof(SWPairView), hipMemcpyHostToDevice);
    hipMemset(d_res, 0, (size_t)N * sizeof(SWResult));

    // 1) the warp-per-row score kernel (the fast path).
    int max_n = 0, max_m = 0;
    for (const auto& c : cases) {
        if ((int)c.pattern.size() > max_n) max_n = (int)c.pattern.size();
        if ((int)c.text.size()    > max_m) max_m = (int)c.text.size();
    }
    const int ints_per_w = genoaligner::sw_smem_ints_per_warp(max_n);
    const int sblock     = genoaligner::SW_WARPS_PER_BLOCK * dev_warp;
    const int sgrid      = (N + genoaligner::SW_WARPS_PER_BLOCK - 1)
                           / genoaligner::SW_WARPS_PER_BLOCK;
    const size_t smem    = (size_t)genoaligner::SW_WARPS_PER_BLOCK
                           * (size_t)ints_per_w * sizeof(int);
    hipLaunchKernelGGL(sw_score_kernel, dim3((unsigned)sgrid), dim3((unsigned)sblock),
                       smem, (hipStream_t)0, d_pairs, P, d_res, N, ints_per_w);

    // 2) the trace kernel (one thread per pair) with its workspaces.
    const size_t dir_stride     = (size_t)(max_m + 1) * (size_t)(max_n + 1);
    const size_t scratch_stride = (size_t)2 * ((size_t)max_n + 1);
    const int    cigar_cap      = max_m + max_n + 4;

    SWTraceResult* d_tr    = nullptr;
    int*           d_cg    = nullptr;
    int*           d_meta  = nullptr;
    uint8_t*       d_dirs  = nullptr;
    int*           d_scr   = nullptr;
    if (hipMalloc(&d_tr,   (size_t)N * sizeof(SWTraceResult))      != hipSuccess ||
        hipMalloc(&d_cg,   (size_t)N * (size_t)cigar_cap * 4)      != hipSuccess ||
        hipMalloc(&d_meta, (size_t)N * 4 * sizeof(int))            != hipSuccess ||
        hipMalloc(&d_dirs, (size_t)N * dir_stride)                 != hipSuccess ||
        hipMalloc(&d_scr,  (size_t)N * scratch_stride * 4)         != hipSuccess) {
        printf("  hipMalloc(trace workspaces) failed: dirs=%zu B\n",
               (size_t)N * dir_stride); return 1;
    }
    hipMemset(d_tr, 0, (size_t)N * sizeof(SWTraceResult));
    printf("  trace workspace: %zu B dirs + %zu B scratch over %d pairs (cap=%d)\n\n",
           (size_t)N * dir_stride, (size_t)N * scratch_stride * 4, N, cigar_cap);

    const int tblock = 64;
    const int tgrid  = (N + tblock - 1) / tblock;
    hipLaunchKernelGGL(sw_trace_kernel, dim3((unsigned)tgrid), dim3((unsigned)tblock),
                       0, (hipStream_t)0, d_pairs, P, d_tr, d_cg, d_meta,
                       d_dirs, dir_stride, d_scr, scratch_stride, N, cigar_cap);

    hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
        printf("  LAUNCH FAILED: %s\n", hipGetErrorString(le)); return 1;
    }
    if (hipDeviceSynchronize() != hipSuccess) {
        printf("  KERNEL FAULT in execution\n"); return 1;
    }

    std::vector<SWResult>      hr((size_t)N);
    std::vector<SWTraceResult> htr((size_t)N);
    std::vector<int>           hmeta((size_t)N * 4);
    std::vector<int>           hcg((size_t)N * (size_t)cigar_cap);
    hipMemcpy(hr.data(),    d_res,  (size_t)N * sizeof(SWResult),      hipMemcpyDeviceToHost);
    hipMemcpy(htr.data(),   d_tr,   (size_t)N * sizeof(SWTraceResult), hipMemcpyDeviceToHost);
    hipMemcpy(hmeta.data(), d_meta, (size_t)N * 4 * sizeof(int),       hipMemcpyDeviceToHost);
    hipMemcpy(hcg.data(),   d_cg,   (size_t)N * (size_t)cigar_cap * 4, hipMemcpyDeviceToHost);

    std::ofstream emit;
    if (emit_path) {
        emit.open(emit_path);
        emit << "idx\ttext\tpattern\tscore\tstart_i\tstart_j\tend_i\tend_j\tcigar\n";
    }

    printf("-- trace kernel vs score kernel vs reference (GPU results) --\n");
    int shown = 0;
    for (int i = 0; i < N; ++i) {
        const RefOut w = ref_sw_trace(cases[(size_t)i].text, cases[(size_t)i].pattern, P);
        const SWTraceResult& r = htr[(size_t)i];
        const int* meta = &hmeta[(size_t)i * 4];
        const int* rev  = &hcg[(size_t)i * (size_t)cigar_cap];
        const int  used = meta[1];
        const std::string cigar = genoaligner::sw_cigar_from_rev(rev, used);
        const std::string refcg = genoaligner::sw_cigar_from_rev(
            w.cigar_rev.empty() ? nullptr : w.cigar_rev.data(), (int)w.cigar_rev.size());

        bool ok = true;
        const char* why = "";
        if (r.score != w.score)                  { ok = false; why = "trace.score != reference"; }
        else if (r.score != hr[(size_t)i].score) { ok = false; why = "trace.score != score-kernel"; }
        else if (r.score == 0) {
            if (used != 0 || r.start_i != -1 || r.end_i != -1)
                { ok = false; why = "score 0 inconsistent"; }
        } else {
            if (meta[3] != 1 || meta[2] != 0) { ok = false; why = "meta"; }
            else if (r.end_i != w.end_i || r.end_j != w.end_j ||
                     r.start_i != w.start_i || r.start_j != w.start_j)
                { ok = false; why = "span != reference"; }
            else if (cigar != refcg) { ok = false; why = "CIGAR != ref walk"; }
            else {
                const genoaligner::SWCigarCheck ck = genoaligner::sw_cigar_check(
                    cigar, cases[(size_t)i].text, cases[(size_t)i].pattern,
                    r.start_i, r.start_j, r.end_i, r.end_j,
                    P.match, P.mismatch, P.gap_open, P.gap_extend, r.score);
                if (!ck.wellformed)  { ok = false; why = "not well-formed"; }
                else if (!ck.rescore_ok) { ok = false; why = "rescore != score"; }
            }
        }
        if (!ok || shown < 8) {
            printf("  %-30s ref=%4d trace=%4d score_k=%4d  cigar=%-14s %s\n",
                   cases[(size_t)i].label.c_str(), w.score, r.score, hr[(size_t)i].score,
                   genoaligner::cigar_run_length(cigar).c_str(),
                   ok ? "ok" : why);
            ++shown;
        }
        if (!ok) ++g_fail;
        if (emit)
            emit << i << '\t' << cases[(size_t)i].text << '\t' << cases[(size_t)i].pattern
                 << '\t' << r.score << '\t' << r.start_i << '\t' << r.start_j
                 << '\t' << r.end_i << '\t' << r.end_j << '\t' << cigar << '\n';
    }
    printf("  (%d/%d agree end to end)\n", N - g_fail, N);

    for (int i = 0; i < N; ++i) { hipFree(d_txt[(size_t)i]); hipFree(d_pat[(size_t)i]); }
    hipFree(d_pairs); hipFree(d_res);
    hipFree(d_tr); hipFree(d_cg); hipFree(d_meta); hipFree(d_dirs); hipFree(d_scr);

    printf("\n");
    if (g_fail == 0) {
        printf("RESULT: PASS -- SW traceback on GPU: score==kernel==reference,\n"
               "        CIGAR well-formed, re-scored, equal to the matrix walk\n");
        return 0;
    }
    printf("RESULT: FAIL (%d)\n", g_fail);
    return 1;
}
