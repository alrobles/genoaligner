// genomsa — multiple sequence alignment on GPU (level-batched profile-profile
// progressive alignment), with a CPU reference mode for verification.
//
//   genomsa input.fasta output.fasta [--cpu] [--kmer K] [--global]
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
        fprintf(stderr, "usage: %s input.fasta output.fasta [--cpu] [--kmer K] [--global]\n",
                argv[0]);
        return 2;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];
    genomsa::Params P;
    bool use_cpu = false;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cpu") use_cpu = true;
        else if (a == "--kmer" && i + 1 < argc) P.kmer_k = atoi(argv[++i]);
        else if (a == "--global") P.free_end_gaps = false;
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
    if (use_cpu) {
        msa = genomsa::msa_align(seqs, P);
    } else {
        genomsa::GpuStats st;
        if (!genomsa::msa_align_gpu(seqs, P, msa, err, &st)) {
            fprintf(stderr, "gpu align failed: %s\n", err.c_str());
            return 1;
        }
        fprintf(stderr,
                "levels=%d pairs=%d dir_peak=%.1fMB | dist=%.2fs tree=%.2fs(%s) align=%.2fs\n",
                st.levels, st.pairs, st.dir_bytes / 1e6,
                st.dist_s, st.tree_s, st.tree_gpu ? "gpu" : "cpu", st.align_s);
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
    return 0;
}
