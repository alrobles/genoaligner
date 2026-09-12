// genoaligner — the example from the README.
//
// This file exists so the README cannot document usage that does not work. It is
// built as part of the test suite (CMake target genoaligner_example) and its
// expected output is printed at the end, so a change that breaks the documented
// usage breaks something visible.
//
// It walks the full path a user takes: read a FASTA, align the pairs you want, write
// the results. No test fixtures, no internal headers -- only the public API and the
// FASTA reader, exactly as a consumer would use them.

#include <genoaligner/api.hpp>
#include <genoaligner/io/fasta.hpp>

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    const char* path = (argc > 1) ? argv[1] : "tests/data/mtdna_human.fa";

    // 1. Read sequences. The reader returns raw bytes: it does not validate that the
    //    sequence is ACGT, because real FASTA carries N, ambiguity codes and
    //    lowercase soft-masking, and rejecting those would be wrong.
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
    if (seq.size() < 400) {
        std::fprintf(stderr, "sequence too short for this example\n");
        return 1;
    }

    // 2. Build the pairs you want aligned. A request carries POINTERS, not copies:
    //    the strings must stay alive until the batch returns.
    struct Window { std::string pattern, text; };
    std::vector<Window> wins;
    for (int i = 0; i < 3; ++i) {
        Window w;
        w.pattern = seq.substr((size_t)(1000 + i * 700), 200);
        w.text    = w.pattern;
        w.text[(size_t)(20 + i * 30)] = (w.text[(size_t)(20 + i * 30)] == 'A') ? 'C' : 'A';
        wins.push_back(std::move(w));
    }

    std::vector<genoaligner::AlignRequest> reqs;
    for (const Window& w : wins) {
        genoaligner::AlignRequest r;
        r.pattern     = w.pattern.data();
        r.pattern_len = (int)w.pattern.size();
        r.text        = w.text.data();
        r.text_len    = (int)w.text.size();
        r.smax        = 64;      // see LIMITS in the README: one bound for the batch,
        r.with_cigar  = true;    // and it is the MINIMUM of the requests'
        reqs.push_back(r);
    }

    // 3. Align. Check status FIRST: a device failure is not the same as "unresolved".
    genoaligner::BatchResult batch = genoaligner::align_batch(reqs);
    if (!batch.ok()) {
        std::fprintf(stderr, "alignment failed: %s\n", batch.error ? batch.error : "?");
        return 1;
    }

    std::printf("aligned %zu pair(s): %d resolved, %d unresolved\n\n",
                batch.results.size(), batch.resolved_count, batch.unresolved_count);

    // 4. Use the results. An unresolved pair is NOT an error: the true distance
    //    exceeded smax. Its score is -1 and its CIGAR is empty.
    for (size_t i = 0; i < batch.results.size(); ++i) {
        const genoaligner::AlignResult& r = batch.results[i];
        if (!r.resolved) {
            std::printf("pair %zu: unresolved (distance > smax)\n", i);
            continue;
        }
        std::printf("pair %zu: score=%d cigar=%s (validated: rescore=%s wellformed=%s)\n",
                    i, r.score, r.cigar.c_str(),
                    r.rescore_ok ? "yes" : "no", r.wellformed_ok ? "yes" : "no");
    }

    std::printf("\nexample complete\n");
    return 0;
}
