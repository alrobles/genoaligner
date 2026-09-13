// genoaligner — Smith-Waterman example.
//
// The SW counterpart of examples/align_fasta.cpp: it walks the same path a user
// takes (read a FASTA, align, consume results) through the PUBLIC API only, so a
// change that breaks documented SW usage breaks something visible. Built as
// genoaligner_example_sw.
//
// SW differs from WFA in the API surface in ways a user must see here:
//  - separate types (SWRequest / SWAlignResult) and entry points
//    (align_sw / align_sw_batch) -- AlignResult.score is an edit DISTANCE,
//    SWAlignResult.score is an alignment SCORE; never mix them
//  - there is no smax: local alignment has no distance budget
//  - results carry local-alignment coordinates (start/end on both sequences)
//  - an empty local alignment is score=0, coordinates=-1, empty CIGAR -- it is
//    resolved, not unresolved

#include <genoaligner/api.hpp>
#include <genoaligner/io/fasta.hpp>

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    const char* path = (argc > 1) ? argv[1] : "tests/data/mtdna_human.fa";

    std::vector<genoaligner::io::FastaRecord> records;
    std::string err;
    if (!genoaligner::io::read_fasta_file(path, &records, &err)) {
        std::fprintf(stderr, "cannot read %s: %s\n", path, err.c_str());
        return 1;
    }

    std::printf("genoaligner %s on %s\n", genoaligner::backend_name(),
                genoaligner::device_name());
    std::printf("read %zu record(s) from %s\n\n", records.size(), path);

    const std::string& seq = records[0].sequence;
    if (seq.size() < 1000) {
        std::fprintf(stderr, "sequence too short for this example\n");
        return 1;
    }

    // Two windows whose best local alignment is a conserved interior block:
    // take a 200-mer, embed it in unrelated flanks, and mutate a few positions.
    // Local alignment finds the conserved block wherever it landed -- WFA would
    // instead report the whole-window edit distance.
    struct Window { std::string pattern, text; };
    std::vector<Window> wins;
    for (int i = 0; i < 3; ++i) {
        Window w;
        std::string core = seq.substr((size_t)(400 + i * 200), 80);
        w.pattern = std::string("TTTTTTTTTT") + core + std::string("TTTTTTTTTT");
        w.text    = std::string("AAAAAAAAAA") + seq.substr((size_t)(400 + i * 200), 80)
                  + std::string("AAAAAAAAAA");
        w.text[(size_t)(15 + i * 7)] = 'N';  // an ambiguity base inside the core
        wins.push_back(std::move(w));
    }

    std::vector<genoaligner::SWRequest> reqs;
    for (const Window& w : wins) {
        genoaligner::SWRequest r;
        r.pattern     = w.pattern.data();
        r.pattern_len = (int)w.pattern.size();
        r.text        = w.text.data();
        r.text_len    = (int)w.text.size();
        r.scoring     = {2, -3, 3, 1};   // match, mismatch, gap_open, gap_extend
        r.with_cigar  = true;
        reqs.push_back(r);
    }

    // Check status FIRST: a device failure is not the same as score=0.
    genoaligner::SWBatchResult batch = genoaligner::align_sw_batch(reqs);
    if (!batch.ok()) {
        std::fprintf(stderr, "SW alignment failed: %s\n", batch.error ? batch.error : "?");
        return 1;
    }

    for (size_t i = 0; i < batch.results.size(); ++i) {
        const genoaligner::SWAlignResult& r = batch.results[i];
        if (!r.resolved) {
            std::printf("pair %zu: not resolved (too_large=%s)\n", i,
                        r.too_large ? "yes" : "no");
            continue;
        }
        if (r.score == 0) {
            std::printf("pair %zu: no positive-scoring local alignment\n", i);
            continue;
        }
        std::printf("pair %zu: score=%d cigar=%s  span text[%d..%d] pattern[%d..%d]"
                    " (validated: rescore=%s wellformed=%s)\n",
                    i, r.score, r.cigar.c_str(), r.start_i, r.end_i, r.start_j, r.end_j,
                    r.rescore_ok ? "yes" : "no", r.wellformed_ok ? "yes" : "no");
    }

    std::printf("\nexample complete\n");
    return 0;
}
