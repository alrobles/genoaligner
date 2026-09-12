// genoaligner — B3: concurrency / reentrancy test.
//
// WHAT THIS MEASURES (and what it deliberately does not)
// -----------------------------------------------------
// The question a caller needs answered is narrow and concrete: can two threads call
// align_batch() at the same time and each get the RIGHT answer? Not "is the library
// thread-safe" in the abstract -- that phrase hides which operations are safe and
// which only happen to work.
//
// Known shared state before this test existed (found by reading, listed so the test
// can be judged against it):
//   - device_name() memoises into a `static char buf[256]` with an unsynchronised
//     check-then-write. Benign in effect (every thread writes the same string) but a
//     data race by the letter of the memory model.
//   - The device itself: hipMalloc/hipFree, launches and hipMemcpy are issued on the
//     default stream unless a stream is passed. Two threads allocating at once is
//     safe at the allocation level (HIP's allocator is synchronised); two threads
//     launching on the SAME default stream concurrently is not -- they interleave.
//
// WHAT THE TEST DOES
//   Runs N threads, each aligning its own batch of REAL mtDNA windows, and checks
//   every thread's results against the CPU DP. A wrong answer under concurrency is
//   the failure that matters, so correctness is asserted per thread, not just "it
//   did not crash" (a crash-free run with wrong numbers is a worse outcome).
//
//   It also runs the same comparison single-threaded first, so a failure can be
//   attributed to concurrency rather than to the data.
//
// If this passes, the honest claim is: "concurrent align_batch() calls on DISJOINT
// inputs returned correct results in this configuration." It is NOT a proof of
// thread-safety, and the doc says so.

#include "include/genoaligner/api.hpp"
#include "include/genoaligner/io/fasta.hpp"
#include "src/reference/edit_distance_cpu.hpp"

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using namespace genoaligner;

static std::atomic<int> g_fail{0};
static std::atomic<int> g_threads_ran{0};
static std::atomic<int> g_pairs_checked{0};

static std::string slice(const std::string& s, size_t from, size_t len)
{
    if (from >= s.size()) return std::string();
    return s.substr(from, std::min(len, s.size() - from));
}

int main()
{
    printf("genoaligner — concurrency test (B3)\n");
    printf("  backend : %s\n", backend_name());
    printf("  device  : %s\n\n", device_name());

    std::vector<io::FastaRecord> hrec;
    std::string err;
    if (!io::read_fasta_file("tests/data/mtdna_human.fa", &hrec, &err)) {
        printf("  cannot read tests/data/mtdna_human.fa: %s\n", err.c_str());
        printf("  (run from the repository root)\n");
        return 2;
    }
    const std::string hum = hrec[0].sequence;
    printf("  human mtDNA: %zu bp\n", hum.size());

    const int NT = 4;          // threads
    const int PER = 12;        // pairs per thread
    printf("  configuration: %d threads x %d pairs\n", NT, PER);

    if (!device_available()) {
        printf("\n  SKIP: no device visible; concurrency NOT verified here.\n");
        printf("RESULT: SKIP -- needs a GPU (exit 77)\n");
        return 77;
    }

    // Build a distinct window set per thread, so no two threads share input.
    std::vector<std::vector<std::string>> pats(NT), texts(NT);
    for (int t = 0; t < NT; ++t) {
        for (int k = 0; k < PER; ++k) {
            const size_t from = (size_t)(t * 700 + k * 130) % (hum.size() - 300);
            const std::string p = slice(hum, from, 120);
            std::string q = p;
            q[(size_t)(k * 7) % q.size()] = (q[(size_t)(k * 7) % q.size()] == 'A') ? 'C' : 'A';
            pats[(size_t)t].push_back(p);
            texts[(size_t)t].push_back(q);
        }
    }

    // ---- baseline: same data, single thread ------------------------------
    printf("\n-- baseline (1 thread) --\n");
    int base_bad = 0;
    for (int t = 0; t < NT; ++t) {
        std::vector<AlignRequest> reqs;
        for (int k = 0; k < PER; ++k) {
            AlignRequest r;
            r.pattern = pats[(size_t)t][(size_t)k].data();
            r.pattern_len = (int)pats[(size_t)t][(size_t)k].size();
            r.text    = texts[(size_t)t][(size_t)k].data();
            r.text_len    = (int)texts[(size_t)t][(size_t)k].size();
            r.smax = 64;
            reqs.push_back(r);
        }
        BatchResult b = align_batch(reqs);
        if (!b.ok()) { printf("  batch failed: %s\n", b.error ? b.error : "?"); ++base_bad; continue; }
        for (int k = 0; k < PER; ++k) {
            const AlignResult& r = b.results[(size_t)k];
            const int want = edit_distance_cpu(
                reqs[(size_t)k].pattern, reqs[(size_t)k].pattern_len,
                reqs[(size_t)k].text,    reqs[(size_t)k].text_len);
            if (!r.resolved) { ++base_bad; continue; }
            if (r.score != want) { ++base_bad; }
            if (!r.rescore_ok || !r.wellformed_ok) { ++base_bad; }
        }
    }
    printf("  single-threaded bad=%d of %d\n", base_bad, NT * PER);
    if (base_bad != 0) {
        printf("  !!! the DATA fails single-threaded; concurrency cannot be judged.\n");
        printf("RESULT: FAIL -- baseline invalid\n");
        return 1;
    }

    // ---- concurrent: NT threads, disjoint inputs, correctness asserted ----
    printf("\n-- concurrent (%d threads) --\n", NT);
    std::vector<std::thread> th;
    std::vector<int> bad_per_thread((size_t)NT, 0);

    for (int t = 0; t < NT; ++t) {
        th.emplace_back([&, t]() {
            std::vector<AlignRequest> reqs;
            for (int k = 0; k < PER; ++k) {
                AlignRequest r;
                r.pattern = pats[(size_t)t][(size_t)k].data();
                r.pattern_len = (int)pats[(size_t)t][(size_t)k].size();
                r.text    = texts[(size_t)t][(size_t)k].data();
                r.text_len    = (int)texts[(size_t)t][(size_t)k].size();
                r.smax = 64;
                reqs.push_back(r);
            }
            BatchResult b = align_batch(reqs);
            if (!b.ok()) { bad_per_thread[(size_t)t] += PER; g_fail.fetch_add(1); return; }
            for (int k = 0; k < PER; ++k) {
                const AlignResult& r = b.results[(size_t)k];
                const int want = edit_distance_cpu(
                    reqs[(size_t)k].pattern, reqs[(size_t)k].pattern_len,
                    reqs[(size_t)k].text,    reqs[(size_t)k].text_len);
                int bad = 0;
                if (!r.resolved)                    bad = 1;
                else if (r.score != want)           bad = 1;
                else if (!r.rescore_ok || !r.wellformed_ok) bad = 1;
                if (bad) { bad_per_thread[(size_t)t] += 1; g_fail.fetch_add(1); }
            }
            g_pairs_checked.fetch_add(PER);
            g_threads_ran.fetch_add(1);
        });
    }
    for (auto& x : th) x.join();

    for (int t = 0; t < NT; ++t)
        printf("  thread %d: bad=%d of %d\n", t, bad_per_thread[(size_t)t], PER);

    printf("\n  threads completed : %d of %d\n", g_threads_ran.load(), NT);
    printf("  pairs checked     : %d\n", g_pairs_checked.load());
    printf("  incorrect results : %d\n", g_fail.load());

    if (g_fail.load() != 0) {
        printf("\nRESULT: FAIL -- concurrent calls returned incorrect results (%d)\n", g_fail.load());
        return 1;
    }
    if (g_threads_ran.load() != NT) {
        printf("\nRESULT: FAIL -- not all threads completed\n");
        return 1;
    }
    printf("\nRESULT: PASS -- %d concurrent calls, %d pairs, all correct\n",
           NT, g_pairs_checked.load());
    printf("  NOTE: this shows correctness holds for CONCURRENT CALLS ON DISJOINT\n");
    printf("        INPUTS in this configuration. It is not a proof of thread-safety,\n");
    printf("        and the documented contract states the limits.\n");
    return 0;
}
