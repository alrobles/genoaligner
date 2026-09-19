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
                        "[--protein|--codon] [--gc-def N] [--gap-open G] [--gap-extend G] "
                        "[--psgp|--no-psgp] [--gappy T|--no-gappy] [--tree-out F] "
                        "[--codon-qc F] [--refine N] [--fs-cost X] [--fs-term-cost X]\n"
                        "                 [--refine-band N] [--local-frame|--no-local-frame]\n",
                argv[0]);
        return 2;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];
    // --protein/--codon must be parsed before the numeric overrides so
    // users can still tune the preset; do a first pass for them.
    genomsa::Params P;
    bool codon_mode = false;
    int gc_def = 1;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--protein") P = genomsa::protein_params();
        else if (a == "--codon") codon_mode = true;
        else if (a == "--gc-def" && i + 1 < argc) gc_def = atoi(argv[i + 1]);
    }
    if (codon_mode) P = genomsa::codon_params(gc_def);
    bool use_cpu = false;
    bool lf_set = false;
    std::string tree_out, qc_out;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cpu") use_cpu = true;
        else if (a == "--protein" || a == "--codon") { /* already applied */ }
        else if (a == "--gc-def" && i + 1 < argc) ++i;
        else if (a == "--kmer" && i + 1 < argc) P.kmer_k = atoi(argv[++i]);
        else if (a == "--global") P.free_end_gaps = false;
        else if (a == "--gap-open" && i + 1 < argc) P.gap_open = atof(argv[++i]);
        else if (a == "--gap-extend" && i + 1 < argc) P.gap_extend = atof(argv[++i]);
        else if (a == "--psgp") P.psgp = true;
        else if (a == "--no-psgp") P.psgp = false;
        else if (a == "--gappy" && i + 1 < argc) P.gappy = atof(argv[++i]);
        else if (a == "--no-gappy") P.gappy = 0.0f;
        else if (a == "--tree-out" && i + 1 < argc) tree_out = argv[++i];
        else if (a == "--codon-qc" && i + 1 < argc) qc_out = argv[++i];
        else if (a == "--refine" && i + 1 < argc) P.codon_refine = atoi(argv[++i]);
        else if (a == "--fs-cost" && i + 1 < argc) P.codon_fs = atof(argv[++i]);
        else if (a == "--fs-term-cost" && i + 1 < argc) P.codon_fs_term = atof(argv[++i]);
        else if (a == "--refine-band" && i + 1 < argc) P.codon_refine_band = atof(argv[++i]);
        else if (a == "--local-frame") { P.codon_local_frame = 1; lf_set = true; }
        else if (a == "--no-local-frame") { P.codon_local_frame = 0; lf_set = true; }
        else if (a == "--fs-enc-cost" && i + 1 < argc) P.codon_fs_enc = atof(argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    // --local-frame is the default in codon mode (downstream RF parity
    // with base + real gain under frameshifts); --no-local-frame opts out.
    if (codon_mode && !lf_set) P.codon_local_frame = 1;

    std::vector<genoaligner::io::FastaRecord> recs;
    auto tr0 = std::chrono::steady_clock::now();
    std::string err;
    if (!genoaligner::io::read_fasta_file(in_path, &recs, &err)) {
        fprintf(stderr, "read %s: %s\n", in_path, err.c_str());
        return 1;
    }
    std::vector<std::string> seqs;
    for (auto& r : recs) seqs.push_back(r.sequence);
    fprintf(stderr, "read %zu records from %s (%.2fs)\n", recs.size(), in_path,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tr0).count());

    // codon mode: tokenize to codons up front, decode on output
    std::vector<genomsa::CodonQc> cqc;
    const std::vector<std::string> orig = codon_mode ? seqs : std::vector<std::string>{};
    if (codon_mode) {
        seqs = P.codon_local_frame
               ? genomsa::codon_encode_local(seqs, P, &cqc)
               : genomsa::codon_encode(seqs, P.gc_def, &cqc);
        int f1 = 0, f2 = 0, st = 0, pt = 0;
        for (auto& q : cqc) { f1 += q.frame == 1; f2 += q.frame == 2;
                              st += q.stops; pt += q.partial; }
        fprintf(stderr, "codon: %zu seqs encoded (gc=%d) | frame1=%d "
                        "frame2=%d  stops=%d  partials=%d\n",
                seqs.size(), P.gc_def, f1, f2, st, pt);
    }

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
                "levels=%d pairs=%d dir_peak=%.1fMB | dist=%.2fs tree=%.2fs(%s) align=%.2fs\n",
                st.levels, st.pairs, st.dir_bytes / 1e6,
                st.dist_s, st.tree_s, st.tree_gpu ? "gpu" : "cpu", st.align_s);
    }
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "align total: %.2fs (%s)\n",
            std::chrono::duration<double>(t1 - t0).count(),
            use_cpu ? "cpu-ref" : "gpu");

    if (codon_mode) {
        msa = genomsa::codon_decode(msa);
        // --local-frame marks fs events as token-64 placeholders; only the
        // refine pass (driven by the raw seqs) restores the real nts, so
        // it is mandatory for content-correct output.
        if (P.codon_local_frame && P.codon_refine < 1) {
            P.codon_refine = 1;
            fprintf(stderr, "note: --local-frame implies --refine 1\n");
        }
        if (P.codon_refine > 0) {
            auto r0 = std::chrono::steady_clock::now();
            bool gpu_ref = false;
            if (!use_cpu) {
                float ks = 0.f;
                gpu_ref = genomsa::codon_refine_gpu(orig, msa, P,
                                                    P.codon_refine, msa,
                                                    err, &ks);
                if (gpu_ref)
                    fprintf(stderr, "codon refine: %d round(s), %.2fs "
                            "(gpu kernel %.2fs)\n", P.codon_refine,
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - r0).count(),
                            (double)ks);
                else
                    fprintf(stderr, "gpu refine failed (%s); host fallback\n",
                            err.c_str());
            }
            if (!gpu_ref) {
                msa = genomsa::codon_refine(orig, msa, P, P.codon_refine);
                fprintf(stderr, "codon refine: %d round(s), %.2fs\n",
                        P.codon_refine,
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - r0).count());
            }
        }
    }

    FILE* f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    for (size_t i = 0; i < recs.size(); ++i)
        fprintf(f, ">%s\n%s\n", recs[i].id.c_str(), msa[i].c_str());
    fclose(f);
    fprintf(stderr, "wrote %s\n", out_path);

    if (!qc_out.empty() && codon_mode) {
        FILE* qf = fopen(qc_out.c_str(), "w");
        if (qf) {
            for (size_t i = 0; i < recs.size(); ++i)
                fprintf(qf, "%s\t%d\t%d\t%d\n", recs[i].id.c_str(),
                        cqc[i].frame, cqc[i].stops, cqc[i].partial);
            fclose(qf);
            fprintf(stderr, "wrote %s\n", qc_out.c_str());
        }
    }

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
