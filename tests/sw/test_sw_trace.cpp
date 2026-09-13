// genoaligner — Smith-Waterman traceback gate (Fase B), CPU side.
//
// WHAT IS UNDER TEST
// ------------------
// sw_trace_kernel: a one-thread-per-pair kernel that recomputes the SW matrix
// serially while recording a direction byte per cell, then walks it backward
// into a CIGAR. The kernel's forward pass is deliberately a DIFFERENT code path
// from sw_score_kernel's warp scan, and this test compares all three:
//
//   score_kernel.score == trace_kernel.score == reference.score
//
// and then, on the CIGAR itself:
//
//   (b) well-formed -- ops consume exactly the aligned span, M on equal chars,
//        X on differing (sw_cigar.hpp; counts alone were insufficient for WFA)
//   (a) re-score    -- score(CIGAR) under the affine scheme == score
//   (R) reference path -- the CIGAR must EQUAL an independent walk that runs on
//        the full H/E/F matrices by VALUE comparison. Both walks use the same
//        tie-breaks (diag > E > F; extend wins only when strictly greater), so
//        equality is the expectation, not a coincidence to excuse.
//
// NEGATIVE CONTROL
// ----------------
// A checker that cannot go red is worth less than none (the WFA lesson: the
// CIGAR never worked through the API while its test passed). Two deliberate
// corruptions must be caught: a CIGAR with one op flipped (M->X) must fail
// wellformed, and a direction table with one mid-path byte cleared to STOP must
// produce a CIGAR that differs from the reference.
//
// --emit <file> writes one TSV row per case for the SeqAn3 oracle:
//   idx \t text \t pattern \t score \t start_i \t start_j \t end_i \t end_j \t cigar

#include "genoaligner/backend/sw_kernel.hip"
#include "genoaligner/backend/sw_kernel_impl.hip"
#include "../parity/cigar.hpp"          // wfa_op_to_char, cigar_run_length
#include "sw_cigar.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <fstream>

#ifdef GENOALIGNER_HIP_SHIM
thread_local uint3 threadIdx{0, 0, 0};
uint3 blockIdx{0, 0, 0};
dim3  blockDim{1, 1, 1};
dim3  gridDim{1, 1, 1};
#endif

using genoaligner::SWParams;
using genoaligner::SWPairView;
using genoaligner::SWResult;
using genoaligner::SWTraceResult;
using genoaligner::sw_score_kernel;
using genoaligner::sw_trace_kernel;

// ---------------------------------------------------------------------------
// Independent reference: full matrices + a value-comparison walk.
// ---------------------------------------------------------------------------
struct RefOut {
    int score = 0;
    int start_i = -1, start_j = -1, end_i = -1, end_j = -1;
    std::vector<int> cigar_rev;          // reversed op codes, same convention
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

    // Walk by VALUE: at each cell, ask which predecessor reproduces the stored
    // score, applying the same priorities the forward pass used (diag > E > F;
    // a gap extends only when it beats opening). A different mechanism than the
    // kernel's byte table -- agreement is evidence, not construction.
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
            } else {                       // h == F
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

// Run sw_trace_kernel for every case through the shim's direct-call mode:
// thread-per-pair means blockIdx*blockDim+threadIdx selects the pair, and the
// "device" buffers are host vectors sized exactly to the launch contract.
struct TraceOut {
    std::vector<SWTraceResult> res;
    std::vector<int>           meta;       // 4 ints per pair
    std::vector<int>           cigars;     // cigar_cap per pair
    int                        cigar_cap = 0;
    size_t                     dir_stride = 0;
    std::vector<uint8_t>       dirs;       // kept for the negative control
};

static TraceOut run_trace(const std::vector<Case>& cases, SWParams prm,
                          size_t dir_stride_override = 0, int cigar_cap_override = 0)
{
    TraceOut t;
    const int N = (int)cases.size();
    size_t max_cells = 0; int max_n = 0, max_m = 0;
    for (const auto& c : cases) {
        const size_t cells = (c.text.size() + 1) * (c.pattern.size() + 1);
        if (cells > max_cells) max_cells = cells;
        if ((int)c.pattern.size() > max_n) max_n = (int)c.pattern.size();
        if ((int)c.text.size()    > max_m) max_m = (int)c.text.size();
    }
    t.dir_stride = dir_stride_override ? dir_stride_override : max_cells;
    const size_t scratch_stride = 2 * ((size_t)max_n + 1);
    t.cigar_cap  = cigar_cap_override ? cigar_cap_override : max_m + max_n + 4;

    t.res.assign((size_t)N, SWTraceResult{});
    t.meta.assign((size_t)N * 4, -1);
    t.cigars.assign((size_t)N * (size_t)t.cigar_cap, -1);
    t.dirs.assign((size_t)N * t.dir_stride, 0xFF);       // poison: unwritten must read as garbage
    std::vector<int> scratch((size_t)N * scratch_stride, 0x7f7f7f7f);

    std::vector<SWPairView> pv((size_t)N);
    for (int i = 0; i < N; ++i) {
        pv[(size_t)i] = SWPairView{cases[(size_t)i].text.data(), (int)cases[(size_t)i].text.size(),
                                   cases[(size_t)i].pattern.data(), (int)cases[(size_t)i].pattern.size()};
    }

    blockDim = dim3{64, 1, 1};
    for (int pid = 0; pid < N; ++pid) {
        blockIdx  = uint3{(unsigned)(pid / 64), 0, 0};
        threadIdx = uint3{(unsigned)(pid % 64), 0, 0};
        sw_trace_kernel(pv.data(), prm, t.res.data(), t.cigars.data(), t.meta.data(),
                        t.dirs.data(), t.dir_stride, scratch.data(), scratch_stride,
                        N, t.cigar_cap);
    }
    blockDim = dim3{1, 1, 1}; blockIdx = uint3{0, 0, 0}; threadIdx = uint3{0, 0, 0};
    return t;
}

// Run the SCORE kernel (warp-per-row) under the warp emulation, so the gate also
// asserts trace.score == score_kernel.score -- the two code paths checking each
// other, which is the whole point of keeping them different.
static std::vector<SWResult> run_score_grid(const std::vector<Case>& cases,
                                            SWParams prm, int wsize)
{
    const int N = (int)cases.size();
    std::vector<SWPairView> pv((size_t)N);
    int max_n = 0;
    for (int i = 0; i < N; ++i) {
        pv[(size_t)i] = SWPairView{cases[(size_t)i].text.data(), (int)cases[(size_t)i].text.size(),
                                   cases[(size_t)i].pattern.data(), (int)cases[(size_t)i].pattern.size()};
        if ((int)cases[(size_t)i].pattern.size() > max_n) max_n = (int)cases[(size_t)i].pattern.size();
    }
    std::vector<SWResult> out((size_t)N, SWResult{-1, -1, -1});
    const int nthreads = genoaligner::SW_WARPS_PER_BLOCK * wsize;
    const int ints_per_w = genoaligner::sw_smem_ints_per_warp(max_n);
    shim::smem_vec().assign((size_t)genoaligner::SW_WARPS_PER_BLOCK * (size_t)ints_per_w, 0);
    warpSize = wsize;
    blockDim = dim3{(unsigned)nthreads, 1, 1};
    const int nblocks = (N + genoaligner::SW_WARPS_PER_BLOCK - 1) / genoaligner::SW_WARPS_PER_BLOCK;
    for (int b = 0; b < nblocks; ++b) {
        blockIdx = uint3{(unsigned)b, 0, 0};
        shim::run_block(nthreads, [&] {
            sw_score_kernel(pv.data(), prm, out.data(), N, ints_per_w);
        });
    }
    blockDim = dim3{1, 1, 1}; blockIdx = uint3{0, 0, 0}; threadIdx = uint3{0, 0, 0};
    return out;
}

int main(int argc, char** argv)
{
    const char* emit_path = nullptr;
    for (int a = 1; a < argc; ++a)
        if (std::strcmp(argv[a], "--emit") == 0 && a + 1 < argc) emit_path = argv[++a];

    printf("genoaligner — SW traceback gate (direction-byte kernel, CPU emulation)\n\n");

    std::vector<Case> cases = {
        {"A", "A", "single base"},
        {"ACGTACGT", "ACGTACGT", "8 identical"},
        {"ACGTACGT", "TGCATGCA", "shares one A"},
        {"AAAAAAAA", "CCCCCCCC", "disjoint -> empty"},
        {"ACGTTTACGT", "ACGTACGT", "gap of 2 in pattern (I run)"},
        {"ACGTACGT", "ACGTTTACGT", "gap of 2 in text (D run)"},
        {"AACGTTTACGTAA", "TTCGTACGTT", "gap inside shifted flanks"},
        {std::string(10,'T') + "ACGTACGTACGT" + std::string(10,'G'),
         std::string(10,'C') + "ACGTACGTACGT" + std::string(10,'A'),
         "core with divergent flanks"},
        {"ACACACAC", "ACAC", "tandem repeat"},
        {"", "", "both empty"},
        {"ACGT", "", "empty pattern"},
        {"", "ACGT", "empty text"},
        // Affine behaviour: a gap long enough that extending beats re-opening
        // must produce one I/D RUN, not scattered singles.
        {std::string("ACGT") + std::string(6, 'T') + "ACGT",
         std::string("ACGT") + "ACGT", "long gap in pattern (6I)"},
        {std::string("ACGT") + "ACGT",
         std::string("ACGT") + std::string(6, 'T') + "ACGT", "long gap in text (6D)"},
        // Two gaps in one alignment.
        {"ACGTAAACGTGGGACGT", "ACGTACGTACGT", "two gaps"},
        {std::string(200,'G') + "ACGTACGT" + std::string(50,'T'),
         std::string(40,'A') + "ACGTACGT" + std::string(90,'C'), "n=170,m=258 core"},
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
        for (int k = 0; k < 96; ++k) {
            const int m = (int)(rng() % 131), n = (int)(rng() % 131);
            std::string t, q;
            for (int i = 0; i < m; ++i) t.push_back(alpha[rng() % 4]);
            for (int i = 0; i < n; ++i) q.push_back(alpha[rng() % 4]);
            cases.push_back({t, q, "random"});
        }
    }

    // Two standard schemes (match,mismatch,gap_open,gap_extend), both with
    // gap_open >= gap_extend. gap_extend > gap_open is NOT gated: under it the
    // recurrence prefers re-opening adjacent 1-gaps over extending, and the
    // resulting path collapses into one gap RUN in the CIGAR -- score and CIGAR
    // then disagree BY CONSTRUCTION (a degenerate regime, like match=0 for the
    // score itself). See sw_kernel.hip's Fase B note; the API will reject it.
    const SWParams schemes[2] = { {1, -1, 2, 1}, {2, -3, 3, 1} };
    int g_fail = 0;

    std::ofstream emit;
    if (emit_path) {
        emit.open(emit_path);
        emit << "idx\ttext\tpattern\tscore\tstart_i\tstart_j\tend_i\tend_j\tcigar\n";
    }

    for (int si = 0; si < 2; ++si) {
        const SWParams P = schemes[si];
        printf("-- scheme %d: match=%d mismatch=%d gap_open=%d gap_extend=%d --\n",
               si, P.match, P.mismatch, P.gap_open, P.gap_extend);

        const int N = (int)cases.size();
        std::vector<RefOut> ref((size_t)N);
        for (int i = 0; i < N; ++i)
            ref[(size_t)i] = ref_sw_trace(cases[(size_t)i].text, cases[(size_t)i].pattern, P);

        TraceOut tr = run_trace(cases, P);
        std::vector<SWResult> sc = run_score_grid(cases, P, 64);

        int fails = 0, shown = 0;
        for (int i = 0; i < N; ++i) {
            const SWTraceResult& r = tr.res[(size_t)i];
            const RefOut&        w = ref[(size_t)i];
            const int*           meta = &tr.meta[(size_t)i * 4];
            const int*           rev  = &tr.cigars[(size_t)i * (size_t)tr.cigar_cap];
            const int            used = meta[1];
            const std::string cigar = genoaligner::sw_cigar_from_rev(rev, used);
            const std::string refcg = genoaligner::sw_cigar_from_rev(
                w.cigar_rev.empty() ? nullptr : w.cigar_rev.data(), (int)w.cigar_rev.size());

            bool ok = true;
            const char* why = "";
            if (r.score != w.score)                 { ok = false; why = "trace.score != reference"; }
            else if (r.score != sc[(size_t)i].score){ ok = false; why = "trace.score != score-kernel"; }
            else if (r.score == genoaligner::SW_TRACE_TOO_BIG) { ok = false; why = "TOO_BIG"; }
            else if (r.score == 0) {
                if (r.start_i != -1 || r.start_j != -1 || r.end_i != -1 || r.end_j != -1)
                    { ok = false; why = "score 0 with non-(-1) coords"; }
                else if (used != 0) { ok = false; why = "score 0 with non-empty CIGAR"; }
            } else {
                if (meta[3] != 1 || meta[2] != 0) { ok = false; why = "meta emitted/truncated"; }
                else if (r.end_i != w.end_i || r.end_j != w.end_j ||
                         r.start_i != w.start_i || r.start_j != w.start_j)
                    { ok = false; why = "span != reference"; }
                else if (cigar != refcg) { ok = false; why = "CIGAR != reference walk"; }
                else {
                    const genoaligner::SWCigarCheck ck = genoaligner::sw_cigar_check(
                        cigar, cases[(size_t)i].text, cases[(size_t)i].pattern,
                        r.start_i, r.start_j, r.end_i, r.end_j,
                        P.match, P.mismatch, P.gap_open, P.gap_extend, r.score);
                    if (!ck.wellformed) { ok = false; why = "not well-formed"; }
                    else if (!ck.rescore_ok) { ok = false; why = "rescore != score"; }
                }
            }
            if (!ok) {
                ++fails;
                if (shown++ < 8)
                    printf("  FAIL [%s] %s  score k=%d ref=%d  cigar=%s ref=%s\n",
                           cases[(size_t)i].label.c_str(), why, r.score, w.score,
                           genoaligner::cigar_run_length(cigar).c_str(),
                           genoaligner::cigar_run_length(refcg).c_str());
            }
            if (emit && si == 0)   // scheme 0 only: the oracle uses one scheme
                emit << i << '\t' << cases[(size_t)i].text << '\t' << cases[(size_t)i].pattern
                     << '\t' << r.score << '\t' << r.start_i << '\t' << r.start_j
                     << '\t' << r.end_i << '\t' << r.end_j << '\t' << cigar << '\n';
        }
        printf("  %d/%d cases: trace == score-kernel == reference, CIGAR == ref walk,\n"
               "  well-formed and re-scored\n\n", N - fails, N);
        g_fail += fails;

        // ---- negative controls -------------------------------------------
        if (si == 0) {
            printf("-- negative control: corrupted inputs must NOT pass --\n");
            // (1) flip one op in a known-good CIGAR: M->X at the first M.
            int nc_bad = 0;
            {
                for (int i = 0; i < N && nc_bad == 0; ++i) {
                    const int* meta = &tr.meta[(size_t)i * 4];
                    if (tr.res[(size_t)i].score <= 0 || meta[1] < 2) continue;
                    const int* rev = &tr.cigars[(size_t)i * (size_t)tr.cigar_cap];
                    std::string cigar = genoaligner::sw_cigar_from_rev(rev, meta[1]);
                    const size_t pos = cigar.find('M');
                    if (pos == std::string::npos) continue;
                    cigar[pos] = 'X';
                    const SWTraceResult& r = tr.res[(size_t)i];
                    const genoaligner::SWCigarCheck ck = genoaligner::sw_cigar_check(
                        cigar, cases[(size_t)i].text, cases[(size_t)i].pattern,
                        r.start_i, r.start_j, r.end_i, r.end_j,
                        P.match, P.mismatch, P.gap_open, P.gap_extend, r.score);
                    if (ck.wellformed && ck.rescore_ok) ++nc_bad;
                }
            }
            printf("  flipped-op CIGARs that still passed: %d (want 0)\n", nc_bad);
            if (nc_bad != 0) ++g_fail;

            // (2) zero one direction byte mid-path of a non-trivial pair: the
            // kernel's walk must then produce a CIGAR that differs from the
            // reference (or breaks wellformed), i.e. the comparison has teeth.
            int nc2_ok = 0;
            {
                for (int i = 0; i < N && nc2_ok == 0; ++i) {
                    const RefOut& w = ref[(size_t)i];
                    if (w.score <= 0 || w.cigar_rev.size() < 4) continue;
                    // Re-run ONE pair with a corrupted dir table.
                    TraceOut t1 = run_trace({cases[(size_t)i]}, P);
                    const int rs = (int)cases[(size_t)i].pattern.size() + 1;
                    // Corrupt the cell in the middle of the aligned path.
                    const int ci = (t1.res[0].start_i + t1.res[0].end_i + 2) / 2;   // ~mid row (1-based)
                    const int cj = (t1.res[0].start_j + t1.res[0].end_j + 2) / 2;
                    t1.dirs[(size_t)ci * rs + cj] = genoaligner::SW_DIR_STOP;
                    int used = 0, si2 = 0, sj2 = 0, ok2 = 1;
                    std::vector<int> cg(t1.cigar_cap, 0);
                    genoaligner::sw_trace_walk(cases[(size_t)i].text.data(), cases[(size_t)i].pattern.data(),
                                               t1.dirs.data(), rs,
                                               t1.res[0].end_i + 1, t1.res[0].end_j + 1,
                                               &si2, &sj2, cg.data(), t1.cigar_cap, &used, &ok2);
                    const std::string c2 = genoaligner::sw_cigar_from_rev(cg.data(), used);
                    const std::string c1 = genoaligner::sw_cigar_from_rev(
                        &t1.cigars[0], t1.meta[1]);
                    if (c2 != c1) ++nc2_ok;
                }
            }
            printf("  dir-corruption runs where walk differed from reference: %d (want >=1)\n", nc2_ok);
            if (nc2_ok == 0) {
                printf("  !!! no case diverged under corruption -- the reference\n"
                       "      comparison may not be exercising the path\n");
                ++g_fail;
            }
            printf("\n");
        }
    }

    if (emit_path) printf("emitted %s\n\n", emit_path);

    if (g_fail == 0) {
        printf("RESULT: PASS -- SW traceback matches an independent matrix walk,\n"
               "        re-scores exactly, and the controls go red when cut\n");
        return 0;
    }
    printf("RESULT: FAIL (%d)\n", g_fail);
    return 1;
}
