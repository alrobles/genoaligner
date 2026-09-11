// genoaligner — Phase 2 parity harness (H2).
//
// THE TEST THAT MATTERS: does the GPU wavefront score match an independent CPU
// reference on a control set of sequence pairs?
//
// Fusion criterion for Phase 2: >=95% agreement on ~1,000 pairs. Below that,
// the formulation is wrong and no amount of optimisation helps.
//
// The control set is generated deterministically (fixed seed) so the pass rate
// is reproducible across machines and runs. Identity levels span the
// phylogenetic regime (50/70/90/100%) plus adversarial cases (empty, all-gap,
// single mismatch) where off-by-one errors in diagonal indexing show up.

#include <hip/hip_runtime.h>

#include "include/genoaligner/backend/wfa_kernel.hip"
#include "src/reference/edit_distance_cpu.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t _e = (call);                                             \
        if (_e != hipSuccess) {                                             \
            fprintf(stderr, "HIP error %s at %s:%d\n",                      \
                    hipGetErrorString(_e), __FILE__, __LINE__);              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

using namespace genoaligner;

struct Case {
    std::string text;
    std::string pattern;
    std::string label;
};

// Deterministic generator: same control set every run, on every machine.
static std::vector<Case> build_control_set(int n_cases, uint32_t seed)
{
    std::vector<Case> cases;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> base(0, 3);
    const char alphabet[] = "ACGT";

    auto mutate = [&](std::string s, int ident_pct) {
        int n_mut = (int)(s.size() * (100 - ident_pct) / 100.0);
        std::uniform_int_distribution<int> pos(0, (int)s.size() - 1);
        std::uniform_int_distribution<int> ch(0, 3);
        for (int i = 0; i < n_mut && !s.empty(); ++i) {
            s[pos(rng)] = alphabet[ch(rng)];
        }
        return s;
    };

    std::uniform_int_distribution<int> chdist(0, 3);

    const int idents[] = {100, 90, 70, 50};
    const int lens[]   = {32, 64, 128, 256};

    for (int c = 0; c < n_cases; ++c) {
        const int ident = idents[c % 4];
        const int len   = lens[(c / 4) % 4];

        std::string text;
        text.reserve(len);
        for (int i = 0; i < len; ++i) text.push_back(alphabet[base(rng)]);

        std::string pattern = mutate(text, ident);
        // Occasionally introduce an indel so gaps are exercised, not just subs.
        if (c % 7 == 0 && pattern.size() > 4) {
            pattern.erase(0, 2);
        } else if (c % 11 == 0 && !pattern.empty()) {
            pattern.insert(pattern.begin() + pattern.size() / 2, alphabet[chdist(rng)]);
        }

        char lab[64];
        snprintf(lab, sizeof(lab), "len=%d ident=%d%%", len, ident);
        cases.push_back({text, pattern, lab});
    }

    // Adversarial edge cases -- these are where off-by-one bugs live.
    cases.push_back({"", "", "both empty"});
    cases.push_back({"ACGT", "", "empty pattern"});
    cases.push_back({"", "ACGT", "empty text"});
    cases.push_back({"A", "T", "single mismatch"});
    cases.push_back({"AAAA", "AAAA", "identical"});
    cases.push_back({"AAAA", "AA", "pattern is prefix"});
    cases.push_back({"ACGTACGT", "TTTTTTTT", "no match at all"});
    return cases;
}

int main(int argc, char** argv)
{
    int n_cases = (argc > 1) ? atoi(argv[1]) : 1000;

    // --emit <path>: write pairs + scores for the external oracle (Fase 3).
    std::string emit_path;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--emit" && i + 1 < argc) {
            emit_path = argv[i + 1];
            ++i;
        }
    }

    const int smax = 64;   // cap on edit distance we will chase

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("genoaligner WFA parity harness (H2)\n");
    printf("  device : %s\n", prop.name);
    printf("  arch   : %s\n", prop.gcnArchName);
    printf("  cases  : %d\n", n_cases);
    printf("  smax   : %d\n", smax);
    if (!emit_path.empty()) printf("  emit   : %s\n", emit_path.c_str());
    printf("\n");

    auto cases = build_control_set(n_cases, 12345u);

    // ---- host-side reference -------------------------------------------------
    std::vector<int> ref(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) {
        ref[i] = edit_distance_cpu(cases[i].pattern.data(), (int)cases[i].pattern.size(),
                                   cases[i].text.data(),    (int)cases[i].text.size());
    }

    // ---- device buffers ------------------------------------------------------
    std::vector<PairView> views(cases.size());
    // Keep the strings alive in one contiguous block so device pointers stay valid.
    std::vector<std::string> text_store, pat_store;
    text_store.reserve(cases.size());
    pat_store.reserve(cases.size());
    for (auto& c : cases) { text_store.push_back(c.text); pat_store.push_back(c.pattern); }
    for (size_t i = 0; i < cases.size(); ++i) {
        views[i].text        = text_store[i].data();
        views[i].text_len    = (int)text_store[i].size();
        views[i].pattern     = pat_store[i].data();
        views[i].pattern_len = (int)pat_store[i].size();
        views[i].max_score   = smax;
    }

    char *d_text = nullptr, *d_pat = nullptr;
    PairView* d_pairs = nullptr;
    int* d_scores = nullptr;

    size_t tot_text = 0, tot_pat = 0;
    for (auto& c : cases) { tot_text += c.text.size() + 1; tot_pat += c.pattern.size() + 1; }
    // Pad so zero-length strings still get a valid (non-null) pointer.
    tot_text += 1; tot_pat += 1;

    HIP_CHECK(hipMalloc(&d_text, tot_text));
    HIP_CHECK(hipMalloc(&d_pat, tot_pat));
    HIP_CHECK(hipMalloc(&d_pairs, cases.size() * sizeof(PairView)));
    HIP_CHECK(hipMalloc(&d_scores, cases.size() * sizeof(int)));

    // Pack strings and fix up pointers to device addresses.
    //
    // FIX (R1 review): the previous version stored host byte-offsets by casting
    // them into the pointer fields ("views[i].text = (const char*)(ot)") and then
    // reinterpreted them back. That is UB, and it is fragile under the kernel's
    // __restrict__ qualifiers. Use an explicit 64-bit offset field instead.
    std::vector<char> h_text(tot_text, 0), h_pat(tot_pat, 0);
    std::vector<uint64_t> text_off(cases.size()), pat_off(cases.size());
    size_t ot = 0, op = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        size_t lt = text_store[i].size(), lp = pat_store[i].size();
        memcpy(h_text.data() + ot, text_store[i].data(), lt);
        memcpy(h_pat.data()  + op, pat_store[i].data(),  lp);
        text_off[i] = (uint64_t)ot;
        pat_off[i]  = (uint64_t)op;
        ot += lt + 1; op += lp + 1;
    }
    HIP_CHECK(hipMemcpy(d_text, h_text.data(), tot_text, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_pat,  h_pat.data(),  tot_pat,  hipMemcpyHostToDevice));

    // Convert offsets to real device pointers, in the host-side view array.
    for (size_t i = 0; i < views.size(); ++i) {
        views[i].text    = d_text + text_off[i];
        views[i].pattern = d_pat  + pat_off[i];
    }
    HIP_CHECK(hipMemcpy(d_pairs, views.data(), views.size() * sizeof(PairView),
                        hipMemcpyHostToDevice));

    // ---- launch -------------------------------------------------------------
    // Blocks must cover 2*smax+1 diagonals in one pass, else the thread-to-
    // diagonal mapping (k = tid - s) silently misses diagonals.
    const int need = 2 * smax + 1;
    int block = 1;
    while (block < need) block <<= 1;
    if (block > 1024) block = 1024;
    // Each wavefront needs padding on BOTH ends, one slot each: the reads of
    // prev[idx(k-1)] at k=-smax and prev[idx(k+1)] at k=+smax must land on a
    // sentinel slot, not out of bounds. Used range is [0, 2*smax+2], so each
    // wavefront occupies 2*smax+3 slots and two ping-pong waves are needed.
    //
    // The +2 is load-bearing, not defensive: sizing this as 2*(2*smax+1) made
    // the kernel's high-side read run one int past the allocation, which on the
    // MI210 is a memory access fault (H2 job 29184154, rc=134).
    const size_t shmem = (size_t)2 * (2 * smax + 3) * sizeof(int);
    printf("  block  : %d threads (need %d for 2*smax+1 diagonals)\n\n", block, need);
    wfa_score_kernel<<<(int)cases.size(), block, shmem>>>(d_pairs, d_scores, smax);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int> got(cases.size());
    HIP_CHECK(hipMemcpy(got.data(), d_scores, cases.size() * sizeof(int),
                        hipMemcpyDeviceToHost));

    // ---- compare ------------------------------------------------------------
    int pass = 0, fail = 0, abandoned = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        if (got[i] < 0) { ++abandoned; continue; }   // unresolved within smax
        if (got[i] == ref[i]) ++pass;
        else {
            ++fail;
            if (fail <= 10) {
                printf("  MISMATCH [%s] gpu=%d cpu=%d  (text=%zu pattern=%zu)\n",
                       cases[i].label.c_str(), got[i], ref[i],
                       cases[i].text.size(), cases[i].pattern.size());
            }
        }
    }

    // ---- emit (Fase 3): write the EXACT pairs and scores the GPU produced ----
    //
    // The external oracle must score the same pairs this run scored. Re-deriving
    // the generator in Python would risk a silent RNG divergence between the
    // C++ mt19937 and Python's, so the harness emits the pairs it actually used.
    // Format: index<TAB>pattern<TAB>text<TAB>gpu<TAB>cpu<TAB>label
    // Sequences are ACGT only, so no escaping is needed.
    if (!emit_path.empty()) {
        FILE* fh = fopen(emit_path.c_str(), "w");
        if (!fh) {
            fprintf(stderr, "ERROR: cannot open emit path %s\n", emit_path.c_str());
            return 1;
        }
        fprintf(fh, "# genoaligner Fase 3 GPU scores (emit)\n");
        fprintf(fh, "# smax=%d block=%d cases=%zu\n", smax, block, cases.size());
        fprintf(fh, "index\tpattern\ttext\tgpu\tcpu\tlabel\n");
        for (size_t i = 0; i < cases.size(); ++i) {
            fprintf(fh, "%zu\t%s\t%s\t%d\t%d\t%s\n",
                    i, cases[i].pattern.c_str(), cases[i].text.c_str(),
                    got[i], ref[i], cases[i].label.c_str());
        }
        fclose(fh);
        printf("  emitted   : %s (%zu cases)\n", emit_path.c_str(), cases.size());
    }

    (void)hipFree(d_text); (void)hipFree(d_pat); (void)hipFree(d_pairs);
    (void)hipFree(d_scores);

    const int resolved = pass + fail;
    const double rate = resolved ? (100.0 * pass / resolved) : 0.0;

    printf("\n=== H2 RESULT (WFA score parity) ===\n");
    printf("  resolved  : %d\n", resolved);
    printf("  pass      : %d\n", pass);
    printf("  fail      : %d\n", fail);
    printf("  abandoned : %d  (edit distance > smax=%d)\n", abandoned, smax);
    printf("  parity    : %.2f%%  (criterion: >= 95%%)\n", rate);

    if (rate >= 95.0) { printf("  H2: PASS\n"); return 0; }
    printf("  H2: FAIL -- examine formulation before optimising\n");
    return 1;
}
