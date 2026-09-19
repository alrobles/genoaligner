// genoaligner — the "align every sequence to a reference" benchmark task.
//
// This is the pairwise shape that phylogenyAI's orthology step (reciprocal
// best hit) and the pipeline's "align to bait" usage actually need: N-1
// independent pairwise alignments, all against one reference sequence. It is
// NOT an MSA task -- this program exists to measure the task genoaligner does,
// next to the same task done with per-pair MAFFT calls.
//
// Usage:  vs_reference <gene.fasta> [--ref <index>] [--emit <tsv>]
//                      [--no-cigar] [--reps <n>] [--scoring m,x,go,ge]
//
// Default scoring {1,-1,2,1} matches the SeqAn3 oracle's hardcoded scheme
// (tests/sw/seqan3_sw_oracle.cpp), which is what --emit exists to feed.
//
// Prints per-rep wall/dev times and a summary; with --emit it writes the TSV
// the SeqAn3 oracle (tests/sw/seqan3_sw_oracle.cpp) can score.

#include <genoaligner/api.hpp>
#include <genoaligner/io/fasta.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <gene.fasta> [--ref i] [--emit tsv] "
                             "[--no-cigar] [--reps n]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    int ref_idx = 0, reps = 7;
    bool with_cigar = true;
    const char* emit = nullptr;
    genoaligner::SWScoring scoring{1, -1, 2, 1};   // the oracle-pinned scheme
    for (int a = 2; a < argc; ++a) {
        if (!std::strcmp(argv[a], "--ref") && a + 1 < argc)      ref_idx = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--emit") && a + 1 < argc) emit = argv[++a];
        else if (!std::strcmp(argv[a], "--no-cigar"))             with_cigar = false;
        else if (!std::strcmp(argv[a], "--reps") && a + 1 < argc) reps = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--scoring") && a + 1 < argc)
            std::sscanf(argv[++a], "%d,%d,%d,%d", &scoring.match, &scoring.mismatch,
                        &scoring.gap_open, &scoring.gap_extend);
    }

    std::vector<genoaligner::io::FastaRecord> recs;
    std::string err;
    if (!genoaligner::io::read_fasta_file(path, &recs, &err)) {
        std::fprintf(stderr, "cannot read %s: %s\n", path, err.c_str());
        return 1;
    }
    printf("genoaligner %s on %s\n", genoaligner::backend_name(), genoaligner::device_name());
    printf("file %s: %zu records, ref = #%d (%s, %zu bp)\n",
           path, recs.size(), ref_idx, recs[(size_t)ref_idx].id.c_str(),
           recs[(size_t)ref_idx].sequence.size());

    const std::string& ref = recs[(size_t)ref_idx].sequence;

    // Build all N-1 pairwise requests: every other sequence vs the reference.
    std::vector<genoaligner::SWRequest> reqs;
    double cells = 0;
    for (size_t i = 0; i < recs.size(); ++i) {
        if ((int)i == ref_idx) continue;
        genoaligner::SWRequest r;
        r.text        = recs[i].sequence.data();
        r.text_len    = (int)recs[i].sequence.size();
        r.pattern     = ref.data();
        r.pattern_len = (int)ref.size();
        r.scoring     = scoring;
        r.with_cigar  = with_cigar;
        reqs.push_back(r);
        cells += (double)r.text_len * (double)r.pattern_len;
    }
    const int NP = (int)reqs.size();
    printf("task: %d pairwise alignments vs reference, %.1f Mcells total\n\n",
           NP, cells / 1e6);

    std::vector<double> walls;
    genoaligner::SWBatchResult last;
    for (int rep = 0; rep < reps; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        last = genoaligner::align_sw_batch(reqs);
        const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
        if (!last.ok()) {
            std::fprintf(stderr, "batch failed (rep %d): %s\n", rep,
                         last.error ? last.error : "?");
            return 1;
        }
        walls.push_back(ms);
        printf("  rep %d: %.2f ms  (%.2f GCUPS)\n", rep, ms, cells / (ms * 1e6));
    }

    // Summary + outcome accounting: how many resolved, how many too_large,
    // how many score 0 (no positive local alignment).
    double mn = 1e300, mx = 0, sum = 0;
    for (double w : walls) { sum += w; mn = w < mn ? w : mn; mx = w > mx ? w : mx; }
    const double mean = sum / (double)walls.size();
    int resolved = 0, zero = 0, big = 0, bad = 0;
    for (const auto& r : last.results) {
        if (!r.resolved) { if (r.too_large) ++big; else ++bad; continue; }
        ++resolved;
        if (r.score == 0) ++zero;
        if (with_cigar && r.score > 0 && (!r.rescore_ok || !r.wellformed_ok)) ++bad;
    }
    printf("\nsummary: mean=%.2f ms  min=%.2f  max=%.2f  spread=%.1f%%\n",
           mean, mn, mx, 100.0 * (mx - mn) / mean);
    printf("  resolved=%d  score0=%d  too_large=%d  invalid=%d\n",
           resolved, zero, big, bad);
    printf("  throughput: %.2f GCUPS  (%.0f pairs/s)\n",
           cells / (mean * 1e6), NP / (mean / 1000.0));

    if (emit) {
        FILE* f = fopen(emit, "w");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", emit); return 1; }
        fprintf(f, "idx\ttext\tpattern\tscore\tstart_i\tstart_j\tend_i\tend_j\tcigar\n");
        int w = 0;
        for (size_t i = 0; i < recs.size(); ++i) {
            if ((int)i == ref_idx) continue;
            const auto& r = last.results[(size_t)w++];
            fprintf(f, "%zu\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%s\n", i,
                    recs[i].sequence.c_str(), ref.c_str(),
                    r.score, r.start_i, r.start_j, r.end_i, r.end_j, r.cigar.c_str());
        }
        fclose(f);
        printf("  emitted %d results to %s\n", w, emit);
    }

    if (bad > 0) { printf("RESULT: FAIL (%d invalid results)\n", bad); return 1; }
    printf("RESULT: PASS\n");
    return 0;
}
