// genoaligner — gene_qc: pairwise QC for protein-coding gene FASTAs.
//
// What it does: aligns every sequence in a gene file to a reference
// (bait) pairwise, then walks each CIGAR to answer the questions the
// phylogenyAI cleaning step needs:
//
//   * Does the record align to this gene's bait at all?
//     (score == 0 / very low -> mislabeled accession or contaminant)
//   * How much of it aligns?          (partial fragments vs full CDS)
//   * Does it carry frameshifts?      (indel runs whose length % 3 != 0)
//   * Does it carry in-frame stops?   (ref-codon-guided translation of
//                                      the aligned span)
//
// The frame is taken from the REFERENCE: ref base j is codon position
// j%3, so the bait must start at a codon boundary (the pipeline baits
// are complete-CDS records). A ref codon translates only when all three
// of its columns pair a text base (a 'D' op leaves the codon incomplete
// and is already counted as a potential frameshift).
//
// Genetic code: --code 1 (standard, default) or --code 2 (vertebrate
// mitochondrial: stops are TAA TAG AGA AGG; TGA is Trp). The four
// mtDNA genes in the Upham set (COI, CYTB, ND1, ND2) need --code 2.
//
// Usage:  gene_qc <seqs.fasta> --ref <ref.fasta> [--code 1|2]
//                 [--emit out.tsv] [--scoring m,x,go,ge]
//                 [--min-cov 0.5] [--min-score 0]
//
// Output TSV columns:
//   acc  len  score  si  sj  ei  ej  cov_text  frameshifts  stops  flag
//
// Flags: OK | NO_ALIGN | LOW_COV | FS=n | STOP=n (FS/STOP may combine).

#include <genoaligner/api.hpp>
#include <genoaligner/io/fasta.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool is_stop(char a, char b, char c, int code)
{
    // a,b,c uppercase ACGT.
    if (a == 'T' && b == 'A' && (c == 'A' || c == 'G')) return true;   // TAA TAG
    if (code == 1 && a == 'T' && b == 'G' && c == 'A') return true;    // TGA
    if (code == 2 && a == 'A' && b == 'G' && (c == 'A' || c == 'G')) return true; // AGA AGG
    return false;
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <seqs.fasta> --ref <ref.fasta> [--code 1|2] [--emit tsv]\n"
            "       [--scoring m,x,go,ge] [--min-cov f] [--min-score n]\n",
            argv[0]);
        return 2;
    }
    const char* seqs_path = argv[1];
    const char* ref_path = nullptr;
    const char* emit = nullptr;
    int code = 1;
    double min_cov = 0.5;
    int min_score = 0;
    genoaligner::SWScoring scoring{1, -1, 2, 1};
    for (int a = 2; a < argc; ++a) {
        if (!std::strcmp(argv[a], "--ref") && a + 1 < argc)      ref_path = argv[++a];
        else if (!std::strcmp(argv[a], "--emit") && a + 1 < argc) emit = argv[++a];
        else if (!std::strcmp(argv[a], "--code") && a + 1 < argc) code = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--min-cov") && a + 1 < argc) min_cov = std::atof(argv[++a]);
        else if (!std::strcmp(argv[a], "--min-score") && a + 1 < argc) min_score = std::atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--scoring") && a + 1 < argc)
            std::sscanf(argv[++a], "%d,%d,%d,%d", &scoring.match, &scoring.mismatch,
                        &scoring.gap_open, &scoring.gap_extend);
    }
    if (!ref_path) { std::fprintf(stderr, "--ref required\n"); return 2; }

    std::vector<genoaligner::io::FastaRecord> recs, rrecs;
    std::string err;
    if (!genoaligner::io::read_fasta_file(seqs_path, &recs, &err) ||
        !genoaligner::io::read_fasta_file(ref_path, &rrecs, &err)) {
        std::fprintf(stderr, "fasta error: %s\n", err.c_str());
        return 1;
    }
    const std::string& ref = rrecs.front().sequence;
    printf("gene_qc: %zu records vs ref %s (%zu bp, code %d) on %s\n",
           recs.size(), rrecs.front().id.c_str(), ref.size(), code,
           genoaligner::backend_name());

    std::vector<genoaligner::SWRequest> reqs(recs.size());
    for (size_t i = 0; i < recs.size(); ++i) {
        reqs[i].text        = recs[i].sequence.data();
        reqs[i].text_len    = (int)recs[i].sequence.size();
        reqs[i].pattern     = ref.data();
        reqs[i].pattern_len = (int)ref.size();
        reqs[i].scoring     = scoring;
        reqs[i].with_cigar  = true;
    }
    genoaligner::SWBatchResult br = genoaligner::align_sw_batch(reqs);
    if (!br.ok()) { std::fprintf(stderr, "batch: %s\n", br.error); return 1; }

    FILE* out = emit ? fopen(emit, "w") : stdout;
    if (!out) { std::fprintf(stderr, "cannot write %s\n", emit); return 1; }
    fprintf(out, "acc\tlen\tscore\tsi\tsj\tei\tej\tcov_text\tframeshifts\tstops\tflag\n");

    int n_ok = 0, n_noalign = 0, n_lowcov = 0, n_fs = 0, n_stop = 0, n_bad = 0;
    for (size_t k = 0; k < recs.size(); ++k) {
        const auto& r = br.results[k];
        const std::string& t = recs[k].sequence;

        int fs = 0, stops = 0;
        double cov = 0.0;
        if (r.resolved && r.score > 0) {
            cov = (double)(r.end_i - r.start_i + 1) / (double)t.size();

            // Walk the CIGAR. i = text pos, j = ref pos (both absolute).
            int i = r.start_i, j = r.start_j;
            char codon[3]; int codon_fill = -1;   // ref codon being built
            for (char op : r.cigar) {
                if (op == 'M' || op == 'X') {
                    int pos = j % 3;
                    if (pos == 0) codon_fill = 0;
                    if (codon_fill == pos) codon[codon_fill++] = t[i];
                    else codon_fill = -1;
                    if (codon_fill == 3) {
                        if (is_stop(codon[0], codon[1], codon[2], code)) ++stops;
                        codon_fill = -1;
                    }
                    ++i; ++j;
                } else if (op == 'I') {
                    ++i;
                } else if (op == 'D') {
                    codon_fill = -1;   // ref base unpaired: codon incomplete
                    ++j;
                }
            }
            // close any trailing indel run for the frameshift count
            // (counted below from a second, simpler pass over run lengths)
            // -- recount frameshifts from run lengths, cleaner:
            fs = 0;
            for (size_t p = 0; p < r.cigar.size();) {
                char o = r.cigar[p];
                if (o != 'I' && o != 'D') { ++p; continue; }
                size_t q = p;
                while (q < r.cigar.size() && r.cigar[q] == o) ++q;
                if ((q - p) % 3 != 0) ++fs;
                p = q;
            }
        }

        std::string flag;
        if (!r.resolved || r.score <= min_score) flag = "NO_ALIGN";
        else {
            if (cov < min_cov) flag = "LOW_COV";
            if (fs > 0)    flag += (flag.empty() ? "" : "+") + std::string("FS=") + std::to_string(fs);
            if (stops > 0) flag += (flag.empty() ? "" : "+") + std::string("STOP=") + std::to_string(stops);
            if (flag.empty()) flag = "OK";
        }

        fprintf(out, "%s\t%zu\t%d\t%d\t%d\t%d\t%d\t%.3f\t%d\t%d\t%s\n",
                recs[k].id.c_str(), t.size(), r.score,
                r.start_i, r.start_j, r.end_i, r.end_j,
                cov, fs, stops, flag.c_str());

        if (flag == "OK") ++n_ok;
        else {
            if (flag.find("NO_ALIGN") != std::string::npos) ++n_noalign;
            if (flag.find("LOW_COV") != std::string::npos)  ++n_lowcov;
            if (flag.find("FS=")    != std::string::npos)   ++n_fs;
            if (flag.find("STOP=")  != std::string::npos)   ++n_stop;
        }
        if (r.resolved && r.score > 0 && (!r.rescore_ok || !r.wellformed_ok)) ++n_bad;
    }

    fprintf(stderr, "summary: %zu seqs | OK=%d NO_ALIGN=%d LOW_COV=%d FS=%d STOP=%d | invalid=%d\n",
            recs.size(), n_ok, n_noalign, n_lowcov, n_fs, n_stop, n_bad);
    if (emit) fclose(out);
    return n_bad ? 1 : 0;
}
