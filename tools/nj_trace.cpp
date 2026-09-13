// nj_trace — instrumented exact Neighbor Joining (CPU) that measures, for
// every merge round, whether the exact NJ cherry lies inside a small
// candidate neighbourhood. This is the empirical pre-requisite of the
// sparse-candidate design in docs/SPEC_NJ_GPU.md (phase NJ1): no kernel is
// written until Recall@k on real genes says the idea is worth it.
//
//   nj_trace input.fasta [--kmer K] [--threads T] [--tree out.tsv]
//            [--rounds out.tsv] [--name GENE]
//
// Semantics are nj_tree's (src/msa/msa_ref.cpp), reproduced here with the
// same summation order and tie break; the resulting tree is checked equal
// to nj_tree(D, n) before any number is reported.
//
// Candidate generators measured per round (m live nodes, cherry (i,j)):
//   fresh   E_k(a) = k nearest live nodes of a by current d (recomputed
//           each round). rank = min(pos of j in E(i), pos of i in E(j)).
//   static  E_k(leaf) = k nearest leaves by the INITIAL D; a cluster's
//           candidates are the union of its leaves' sets (never rebuilt).
//           rank = min over leaves l in i, l' in j of pos(l' in E(l)),
//           and symmetric.
// Certification (fresh, row-wise bound from SPEC §2/NJ1): the round is
// certifiable at k iff for every live a,
//   (m-2)*dk_a - r_a - r_max >= Q_best, dk_a = k-th smallest d_ab (b!=a).
//
// One TSV line per gene on stdout (header with --header).
#include <genoaligner/msa/msa.hpp>
#include <genoaligner/io/fasta.hpp>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace genomsa;

static const int KS[] = {8, 16, 32, 64, 128};
static const int NK = 5;
static const int KMAX = 128;

struct Round {
    int m, i, j;                 // live count, cherry (node ids)
    int rank_fresh, rank_static; // KMAX if outside the top-KMAX
    int cert_k;                  // smallest k in KS certifiable, -1 none
    bool tie;                    // second-best Q within 4 ulp of best
};

static void par_rows(int total, int nthreads,
                     const std::function<void(int, int)>& fn) {
    nthreads = std::max(1, std::min(nthreads, total));
    if (nthreads == 1) { fn(0, total); return; }
    std::vector<std::thread> ts;
    const int chunk = (total + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; ++t) {
        const int lo = t * chunk, hi = std::min(total, lo + chunk);
        if (lo >= hi) break;
        ts.emplace_back([&fn, lo, hi] { fn(lo, hi); });
    }
    for (auto& th : ts) th.join();
}

// Position of `target` in the ascending (d, index) order of row `a` over
// `others`, truncated to KMAX; also returns the k-th smallest distances.
struct RowTop {
    std::vector<int> top;      // up to KMAX node ids, ascending (d, id)
    double dk[NK];             // k-th smallest d for each KS (DBL_MAX if m-1<k)
};

static RowTop row_top(const std::vector<std::vector<double>>& d, int a,
                      const std::vector<int>& alive) {
    RowTop rt;
    std::vector<std::pair<double, int>> v;
    v.reserve(alive.size());
    for (int b : alive) if (b != a) v.push_back({d[a][b], b});
    const int take = std::min<int>(KMAX, (int)v.size());
    if (take < (int)v.size())
        std::nth_element(v.begin(), v.begin() + take, v.end());
    std::sort(v.begin(), v.begin() + take);
    rt.top.resize(take);
    for (int p = 0; p < take; ++p) rt.top[p] = v[p].second;
    for (int q = 0; q < NK; ++q)
        rt.dk[q] = (KS[q] <= take) ? v[KS[q] - 1].first : DBL_MAX;
    return rt;
}

static int pos_in(const std::vector<int>& top, int x) {
    for (int p = 0; p < (int)top.size(); ++p) if (top[p] == x) return p;
    return KMAX;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s input.fasta [--kmer K] [--threads T] "
                        "[--tree out.tsv] [--rounds out.tsv] [--name G] [--header]\n",
                argv[0]);
        return 2;
    }
    const char* in_path = argv[1];
    int kmer = 5;
    int threads = (int)std::thread::hardware_concurrency();
    const char* tree_out = nullptr;
    const char* rounds_out = nullptr;
    std::string name = in_path;
    bool header = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--kmer" && i + 1 < argc) kmer = atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = atoi(argv[++i]);
        else if (a == "--tree" && i + 1 < argc) tree_out = argv[++i];
        else if (a == "--rounds" && i + 1 < argc) rounds_out = argv[++i];
        else if (a == "--name" && i + 1 < argc) name = argv[++i];
        else if (a == "--header") header = true;
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

    auto tA = std::chrono::steady_clock::now();
    std::vector<float> Dp = kmer_distances_mt(seqs, kmer, threads);
    auto tB = std::chrono::steady_clock::now();
    Tree ref = nj_tree_mt(Dp, n, threads);
    auto tC = std::chrono::steady_clock::now();
    const double dist_s = std::chrono::duration<double>(tB - tA).count();
    const double nj_s   = std::chrono::duration<double>(tC - tB).count();

    // ---- dense matrix, identical layout to nj_tree ---------------------
    const int NN = 2 * n - 1;
    auto idx = [n](int i, int j) {
        if (i > j) std::swap(i, j);
        return (size_t)i * n - (size_t)i * (i + 1) / 2 + (j - i - 1);
    };
    std::vector<std::vector<double>> d(NN, std::vector<double>(NN, 0));
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            d[i][j] = d[j][i] = Dp[idx(i, j)];

    // static leaf graph from the initial D
    std::vector<int> leaves_all(n);
    for (int i = 0; i < n; ++i) leaves_all[i] = i;
    std::vector<std::vector<int>> leaf_top(n);
    par_rows(n, threads, [&](int lo, int hi) {
        for (int a = lo; a < hi; ++a) leaf_top[a] = row_top(d, a, leaves_all).top;
    });
    std::vector<int> cluster_of(n);            // leaf -> live node id
    std::vector<std::vector<int>> leaves_in(NN); // node -> its leaves
    for (int i = 0; i < n; ++i) { cluster_of[i] = i; leaves_in[i] = {i}; }

    Tree t;
    t.nodes.resize(n);
    std::vector<int> alive(n);
    for (int i = 0; i < n; ++i) alive[i] = i;
    std::vector<Round> rounds;
    rounds.reserve(n);

    auto tD = std::chrono::steady_clock::now();
    while (alive.size() > 2) {
        const int m = (int)alive.size();
        std::vector<double> r(m, 0);
        for (int a = 0; a < m; ++a)
            for (int b = 0; b < m; ++b) if (a != b)
                r[a] += d[alive[a]][alive[b]];
        double best = 0, second = DBL_MAX; int bi = -1, bj = -1;
        for (int a = 0; a < m; ++a)
            for (int b = a + 1; b < m; ++b) {
                double q = (m - 2) * d[alive[a]][alive[b]] - r[a] - r[b];
                if (bi < 0 || q < best) { second = (bi < 0) ? DBL_MAX : best;
                                          best = q; bi = a; bj = b; }
                else if (q < second) second = q;
            }
        const int xi = alive[bi], xj = alive[bj];
        Round R;
        R.m = m; R.i = xi; R.j = xj;
        R.tie = (second != DBL_MAX) &&
                (second - best) <= 4.0 * DBL_EPSILON * std::max(1.0, std::fabs(best));

        // fresh top-k of both endpoints + certification bound over all rows
        std::vector<RowTop> tops(m);
        par_rows(m, threads, [&](int lo, int hi) {
            for (int a = lo; a < hi; ++a) tops[a] = row_top(d, alive[a], alive);
        });
        R.rank_fresh = std::min(pos_in(tops[bi].top, xj), pos_in(tops[bj].top, xi));
        double rmax = 0;
        for (int a = 0; a < m; ++a) rmax = std::max(rmax, r[a]);
        R.cert_k = -1;
        for (int q = 0; q < NK && R.cert_k < 0; ++q) {
            bool ok = true;
            for (int a = 0; a < m && ok; ++a) {
                if (tops[a].dk[q] == DBL_MAX) continue;   // row has < k others: exact
                ok = (m - 2) * tops[a].dk[q] - r[a] - rmax >= best;
            }
            if (ok) R.cert_k = KS[q];
        }

        // static graph rank
        int rs = KMAX;
        for (int l : leaves_in[xi]) {
            const auto& tp = leaf_top[l];
            for (int p = 0; p < (int)tp.size() && p < rs; ++p)
                if (cluster_of[tp[p]] == xj) { rs = p; break; }
        }
        for (int l : leaves_in[xj]) {
            const auto& tp = leaf_top[l];
            for (int p = 0; p < (int)tp.size() && p < rs; ++p)
                if (cluster_of[tp[p]] == xi) { rs = p; break; }
        }
        R.rank_static = rs;
        rounds.push_back(R);

        // merge exactly as nj_tree
        const int u = (int)t.nodes.size();
        t.nodes.push_back({xi, xj});
        std::vector<int> rest;
        for (int a = 0; a < m; ++a) if (a != bi && a != bj) rest.push_back(alive[a]);
        alive = rest;
        for (int v : rest) d[u][v] = d[v][u] = 0.5 * (d[xi][v] + d[xj][v] - d[xi][xj]);
        alive.push_back(u);
        leaves_in[u] = leaves_in[xi];
        leaves_in[u].insert(leaves_in[u].end(), leaves_in[xj].begin(), leaves_in[xj].end());
        for (int l : leaves_in[u]) cluster_of[l] = u;
    }
    t.root = (int)t.nodes.size();
    t.nodes.push_back({alive[0], alive[1]});
    auto tE = std::chrono::steady_clock::now();
    const double trace_s = std::chrono::duration<double>(tE - tD).count();

    // ---- gate: instrumented NJ == reference NJ -------------------------
    bool same = (t.root == ref.root) && (t.nodes.size() == ref.nodes.size());
    for (size_t k = 0; same && k < t.nodes.size(); ++k)
        same = t.nodes[k].left == ref.nodes[k].left &&
               t.nodes[k].right == ref.nodes[k].right;
    if (!same) {
        fprintf(stderr, "%s: TREE MISMATCH vs nj_tree_mt -- trace invalid\n", name.c_str());
        return 3;
    }

    if (tree_out) {
        FILE* f = fopen(tree_out, "w");
        if (!f) { fprintf(stderr, "cannot write %s\n", tree_out); return 1; }
        fprintf(f, "node\tleft\tright\n");
        for (size_t k = 0; k < t.nodes.size(); ++k)
            fprintf(f, "%zu\t%d\t%d\n", k, t.nodes[k].left, t.nodes[k].right);
        fclose(f);
    }
    if (rounds_out) {
        FILE* f = fopen(rounds_out, "w");
        if (!f) { fprintf(stderr, "cannot write %s\n", rounds_out); return 1; }
        fprintf(f, "round\tm\ti\tj\trank_fresh\trank_static\tcert_k\ttie\n");
        for (size_t k = 0; k < rounds.size(); ++k) {
            const Round& R = rounds[k];
            fprintf(f, "%zu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", k, R.m, R.i, R.j,
                    R.rank_fresh, R.rank_static, R.cert_k, (int)R.tie);
        }
        fclose(f);
    }

    // ---- summary ---------------------------------------------------------
    const double nr = (double)rounds.size();
    int ties = 0;
    double rf[NK] = {0}, rs[NK] = {0}, cert[NK] = {0};
    long sum_rf = 0, sum_rs = 0; int max_rf = 0, max_rs = 0;
    for (const Round& R : rounds) {
        ties += R.tie;
        sum_rf += R.rank_fresh; sum_rs += R.rank_static;
        max_rf = std::max(max_rf, R.rank_fresh); max_rs = std::max(max_rs, R.rank_static);
        for (int q = 0; q < NK; ++q) {
            rf[q]   += R.rank_fresh  < KS[q];
            rs[q]   += R.rank_static < KS[q];
            cert[q] += (R.cert_k >= 0 && R.cert_k <= KS[q]);
        }
    }
    if (header) {
        printf("gene\tn\trounds\tdist_s\tnj_s\ttrace_s\tties");
        for (int q = 0; q < NK; ++q) printf("\tfresh@%d", KS[q]);
        for (int q = 0; q < NK; ++q) printf("\tstatic@%d", KS[q]);
        for (int q = 0; q < NK; ++q) printf("\tcert@%d", KS[q]);
        printf("\tmean_rank_fresh\tmax_rank_fresh\tmean_rank_static\tmax_rank_static\n");
    }
    printf("%s\t%d\t%zu\t%.2f\t%.2f\t%.2f\t%d", name.c_str(), n, rounds.size(),
           dist_s, nj_s, trace_s, ties);
    for (int q = 0; q < NK; ++q) printf("\t%.4f", rf[q] / nr);
    for (int q = 0; q < NK; ++q) printf("\t%.4f", rs[q] / nr);
    for (int q = 0; q < NK; ++q) printf("\t%.4f", cert[q] / nr);
    printf("\t%.2f\t%d\t%.2f\t%d\n", sum_rf / nr, max_rf, sum_rs / nr, max_rs);
    return 0;
}
