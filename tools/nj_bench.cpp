// nj_bench: NJ0 on a real gene -- parity vs nj_tree_mt + timing breakdown.
//
//   nj_bench gene.fasta [--kmer K] [--threads T] [--name G] [--header] [--no-cpu]
//
// One TSV row on stdout:
//   gene n dist_s nj_cpu_s nj_gpu_s upload_s rounds_s download_s dev_MB parity
// parity is PASS/FAIL (Tree.nodes identical) or SKIP with --no-cpu. Diagnostics
// go to stderr. Exit 0 on PASS/SKIP, 1 on FAIL or device error.
//
// Build (device):
//   hipcc -O2 -std=c++17 -I. -Iinclude -o nj_bench tools/nj_bench.cpp
//         src/msa/nj_gpu.cpp src/msa/msa_ref.cpp
#include <genoaligner/msa/msa.hpp>
#include <genoaligner/io/fasta.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using namespace genomsa;

static bool same_tree(const Tree& a, const Tree& b) {
    if (a.root != b.root || a.nodes.size() != b.nodes.size()) return false;
    for (size_t i = 0; i < a.nodes.size(); ++i)
        if (a.nodes[i].left != b.nodes[i].left || a.nodes[i].right != b.nodes[i].right)
            return false;
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s gene.fasta [--kmer K] [--threads T] [--name G] "
                        "[--header] [--no-cpu]\n", argv[0]);
        return 2;
    }
    const char* in_path = argv[1];
    int kmer = 5;
    int threads = (int)std::thread::hardware_concurrency();
    std::string name = in_path;
    bool header = false, run_cpu = true;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--kmer" && i + 1 < argc) kmer = atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = atoi(argv[++i]);
        else if (a == "--name" && i + 1 < argc) name = argv[++i];
        else if (a == "--header") header = true;
        else if (a == "--no-cpu") run_cpu = false;
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (threads < 1) threads = 1;

    std::vector<genoaligner::io::FastaRecord> recs;
    std::string err;
    if (!genoaligner::io::read_fasta_file(in_path, &recs, &err)) {
        fprintf(stderr, "read %s: %s\n", in_path, err.c_str());
        return 1;
    }
    std::vector<std::string> seqs;
    for (auto& r : recs) seqs.push_back(r.sequence);
    const int n = (int)seqs.size();
    if (n < 3) { fprintf(stderr, "need n>=3 (got %d)\n", n); return 1; }

    using clock = std::chrono::steady_clock;
    auto secs = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    auto tA = clock::now();
    std::vector<float> Dp = kmer_distances_mt(seqs, kmer, threads);
    auto tB = clock::now();

    NjStats st;
    Tree gpu = nj_tree_gpu(Dp, n, err, &st);
    auto tC = clock::now();
    if (!err.empty()) {
        fprintf(stderr, "%s: nj_tree_gpu: %s\n", name.c_str(), err.c_str());
        return 1;
    }

    double cpu_s = 0;
    const char* parity = "SKIP";
    if (run_cpu) {
        auto tD = clock::now();
        Tree ref = nj_tree_mt(Dp, n, threads);
        auto tE = clock::now();
        cpu_s = secs(tD, tE);
        parity = same_tree(ref, gpu) ? "PASS" : "FAIL";
    }

    if (header)
        printf("gene\tn\tdist_s\tnj_cpu_s\tnj_gpu_s\tupload_s\trounds_s\tdownload_s\tdev_MB\tparity\n");
    printf("%s\t%d\t%.3f\t%.3f\t%.3f\t%.4f\t%.3f\t%.4f\t%.1f\t%s\n",
           name.c_str(), n, secs(tA, tB), cpu_s, secs(tB, tC),
           st.upload_s, st.rounds_s, st.download_s, st.dev_bytes / 1e6, parity);
    fflush(stdout);
    fprintf(stderr, "%s: n=%d rounds=%d nj_cpu(%dT)=%.3fs nj_gpu=%.3fs "
                    "(up %.4f, rounds %.3f, down %.4f) dev=%.1fMB %s\n",
            name.c_str(), n, st.rounds, threads, cpu_s, secs(tB, tC),
            st.upload_s, st.rounds_s, st.download_s, st.dev_bytes / 1e6, parity);
    return parity[0] == 'F' ? 1 : 0;
}
