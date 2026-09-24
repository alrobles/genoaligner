// genoaligner — multiple sequence alignment example.
//
// The MSA counterpart of examples/align_fasta.cpp and align_sw.cpp: it walks
// the path a user takes (read a FASTA, align the whole record set, consume
// results) through the PUBLIC API only, so a change that breaks documented
// MSA usage breaks something visible. Built as genoaligner_example_msa.
//
// MSA differs from the pairwise entries in ways a user must see here:
//  - the input is a SET of sequences, not a pair; output rows come back
//    equal-length and in input order
//  - MsaMode selects the alphabet: dna (default), protein, codon
//  - codon mode aligns whole codons (MACSE-class): the demo below forces it
//    on a DNA FASTA via --codon and reports the per-sequence encode QC --
//    which is how a caller sees "this record was not a clean CDS" instead
//    of finding out downstream

#include <genoaligner/api.hpp>
#include <genoaligner/io/fasta.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    const char* path = "tests/data/mtdna_human.fa";
    genoaligner::MsaMode mode = genoaligner::MsaMode::dna;
    int gc_def = 1;
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--codon"))   mode = genoaligner::MsaMode::codon;
        else if (!std::strcmp(argv[i], "--protein")) mode = genoaligner::MsaMode::protein;
        else if (!std::strcmp(argv[i], "--gc") && i + 1 < argc) gc_def = std::atoi(argv[++i]);
        else path = argv[i];
    }

    std::vector<genoaligner::io::FastaRecord> records;
    std::string err;
    if (!genoaligner::io::read_fasta_file(path, &records, &err)) {
        std::fprintf(stderr, "cannot read %s: %s\n", path, err.c_str());
        return 1;
    }
    if (records.empty()) {
        std::fprintf(stderr, "%s: no records\n", path);
        return 1;
    }

    genoaligner::MsaRequest req;
    req.mode = mode;
    req.gc_def = gc_def;
    for (const auto& r : records) req.seqs.push_back(r.sequence);

    genoaligner::MsaResult res = genoaligner::msa_align(req);
    if (!res.ok()) {
        std::fprintf(stderr, "msa_align failed: %s\n",
                     res.error ? res.error : "unknown");
        return 1;
    }

    std::printf("aligned %zu records from %s (mode=%s, gc=%d) -> width %d\n\n",
                records.size(), path,
                mode == genoaligner::MsaMode::dna    ? "dna" :
                mode == genoaligner::MsaMode::protein ? "protein" : "codon",
                gc_def, res.width);

    const bool codon = mode == genoaligner::MsaMode::codon;
    for (size_t i = 0; i < records.size(); ++i) {
        std::printf(">%s", records[i].id.c_str());
        if (codon)
            std::printf(" [frame=%d stops=%d partial=%d]",
                        res.qc[i].frame, res.qc[i].stops, res.qc[i].partial);
        std::printf("\n%s\n", res.aligned[i].c_str());
    }
    return 0;
}
