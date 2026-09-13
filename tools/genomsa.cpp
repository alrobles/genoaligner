// genomsa — multiple sequence alignment on GPU (level-batched profile-profile
// progressive alignment), with a CPU reference mode for verification.
//
//   genomsa input.fasta output.fasta [--cpu] [--kmer K] [--global]
//          [--psgp] [--gappy T]
//
// The output rows are written in INPUT record order (the driver restores it
// from the guide-tree merge order). --cpu runs the M1 reference on host; the
// default path launches msa_pp_trace_kernel per guide-tree level.
//
// Build (cluster):  hipcc -O2 -std=c++17 -Iinclude \
//                     tools/genomsa.cpp src/msa/msa_gpu.cpp src/msa/msa_ref.cpp
#include <genoaligner/msa/msa.hpp>
#include <genoaligner/io/fasta.hpp>
#include <cstdio>
#include <chrono>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s input.fasta output.fasta [--cpu] [--kmer K] [--global] "
                        "[--protein] [--gap-open G] [--gap-extend G] "
                        "[--psgp|--no-psgp] [--gappy T|--no-gappy] [--tree-out F]\n",
                argv[0]);
        return 2;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];
    // --protein must be parsed before the numeric overrides so users can
    // still tune the preset; do a first pass for it.
    genomsa::Params P;
    for (int i = 3; i < argc; ++i)
        if (std::string(argv[i]) == "--protein") P = genomsa::protein_params();
    bool use_cpu = false;
    std::string tree_out;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cpu") use_cpu = true;
        else if (a == "--protein") { /* already applied */ }
        else if (a == "--kmer" && i + 1 < argc) P.kmer_k = atoi(argv[++i]);
        else if (a == "--global") P.free_end_gaps = false;
        else if (a == "--gap-open" && i + 1 < argc) P.gap_open = atof(argv[++i]);
        else if (a == "--gap-extend" && i + 1 < argc) P.gap_extend = atof(argv[++i]);
        else if (a == "--psgp") P.psgp = true;
        else if (a == "--no-psgp") P.psgp = false;
        else if (a == "--gappy" && i + 1 < argc) P.gappy = atof(argv[++i]);
        else if (a == "--no-gappy") P.gappy = 0.0f;
        else if (a == "--tree-out" && i + 1 < argc) tree_out = argv[++i];
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }

    std::vector<genoaligner::io::FastaRecord> recs;
    std::string err;
    if (!genoaligner::io::read_fasta_file(in_path, &recs, &err)) {
        fprintf(stderr, "read %s: %s\n", in_path, err.c_str());
        return 1;
    }
    std::vector<std::string> seqs;
    for (auto& r : recs) seqs.push_back(r.sequence);
    fprintf(stderr, "read %zu records from %s\n", recs.size(), in_path);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::string> msa;
    genomsa::Tree guide;
    if (use_cpu) {
        msa = genomsa::msa_align(seqs, P, tree_out.empty() ? nullptr : &guide);
    } else {
        genomsa::GpuStats st;
        if (!genomsa::msa_align_gpu(seqs, P, msa, err, &st,
                                    tree_out.empty() ? nullptr : &guide)) {
            fprintf(stderr, "gpu align failed: %s\n", err.c_str());
            return 1;
        }
        fprintf(stderr,
                "levels=%d pairs=%d dir_peak=%.1fMB | dist=%.2fs tree=%.2fs align=%.2fs\n",
                st.levels, st.pairs, st.dir_bytes / 1e6,
                st.dist_s, st.tree_s, st.align_s);
    }
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "align total: %.2fs (%s)\n",
            std::chrono::duration<double>(t1 - t0).count(),
            use_cpu ? "cpu-ref" : "gpu");

    FILE* f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    for (size_t i = 0; i < recs.size(); ++i)
        fprintf(f, ">%s\n%s\n", recs[i].id.c_str(), msa[i].c_str());
    fclose(f);
    fprintf(stderr, "wrote %s\n", out_path);

    if (!tree_out.empty()) {
        std::vector<std::string> names;
        for (auto& r : recs) names.push_back(r.id);
        FILE* tf = fopen(tree_out.c_str(), "w");
        if (!tf) { fprintf(stderr, "cannot write %s\n", tree_out.c_str()); return 1; }
        std::string nwk = genomsa::tree_to_newick(guide, names);
        fputs(nwk.c_str(), tf);
        fclose(tf);
        fprintf(stderr, "wrote %s\n", tree_out.c_str());
    }
    return 0;
}
