// CPU reference implementation of the genoaligner MSA engine.
// Semantics are fixed here BEFORE the GPU kernel exists -- every later
// kernel is validated against this file's outputs, not the other way.
#include <genoaligner/msa/msa.hpp>
#include <unordered_map>
#include <climits>
#include <cassert>
#include <cstring>
#include <functional>
#include <thread>

namespace genomsa {

// ---------------------------------------------------------- IUPAC table
// Each DNA letter contributes fractional counts to the ACGT subset it
// represents (1/|set| each). Returns false for '-' or unknown letters.
static bool iupac_counts(char c, float cnt[4]) {
    cnt[0] = cnt[1] = cnt[2] = cnt[3] = 0;
    int mask = 0;
    switch (c) {
        case 'A': case 'a': mask = 1;    break;
        case 'C': case 'c': mask = 2;    break;
        case 'G': case 'g': mask = 4;    break;
        case 'T': case 't': case 'U': case 'u': mask = 8; break;
        case 'R': mask = 1|4;            break;
        case 'Y': mask = 2|8;            break;
        case 'M': mask = 1|2;            break;
        case 'K': mask = 4|8;            break;
        case 'S': mask = 2|4;            break;
        case 'W': mask = 1|8;            break;
        case 'H': mask = 1|2|8;          break;
        case 'B': mask = 2|4|8;          break;
        case 'V': mask = 1|2|4;          break;
        case 'D': mask = 1|4|8;          break;
        case 'N': case 'n': mask = 15;   break;
        case 'X': mask = 15;             break;
        default: return false;
    }
    int pop = __builtin_popcount(mask);
    float f = 1.0f / pop;
    for (int b = 0; b < 4; ++b) if (mask & (1 << b)) cnt[b] = f;
    return true;
}

// Amino-acid index in BLOSUM order ARNDCQEGHILKMFPSTWYV.
static int aa_index(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'R': case 'r': return 1;
        case 'N': case 'n': return 2;
        case 'D': case 'd': return 3;
        case 'C': case 'c': return 4;
        case 'Q': case 'q': return 5;
        case 'E': case 'e': return 6;
        case 'G': case 'g': return 7;
        case 'H': case 'h': return 8;
        case 'I': case 'i': return 9;
        case 'L': case 'l': return 10;
        case 'K': case 'k': return 11;
        case 'M': case 'm': return 12;
        case 'F': case 'f': return 13;
        case 'P': case 'p': return 14;
        case 'S': case 's': return 15;
        case 'T': case 't': return 16;
        case 'W': case 'w': return 17;
        case 'Y': case 'y': return 18;
        case 'V': case 'v': return 19;
        default: return -1;
    }
}

// Each letter contributes fractional counts to the symbol subset it
// represents (1/|set| each). alpha==4 -> IUPAC DNA; alpha==20 ->
// amino acids (B: N/D, Z: Q/E, J: I/L, X/O: uniform, U -> C).
// Returns false for '-' or unmapped letters (counted as gap).
static bool sym_counts(char c, float* cnt, int alpha) {
    if (alpha == 4) return iupac_counts(c, cnt);
    for (int i = 0; i < alpha; ++i) cnt[i] = 0;
    if (alpha == 65) {
        // codon-encoded byte = 128 + index (0..63; 64 = other). The +128
        // offset is load-bearing: raw index 45 collides with '-' (gap)
        // in aligned rows.
        int s = (unsigned char)c;
        if (s < 128 || s > 192) return false;
        cnt[s - 128] = 1.f;
        return true;
    }
    switch (c) {
        case 'B': cnt[2] = cnt[3] = 0.5f;      return true;
        case 'Z': cnt[5] = cnt[6] = 0.5f;      return true;
        case 'J': cnt[9] = cnt[10] = 0.5f;     return true;
        case 'U': case 'u': cnt[4] = 1.f;      return true;
        case 'X': case 'x': case 'O': case 'o':
            for (int i = 0; i < alpha; ++i) cnt[i] = 1.f / alpha;
            return true;
        default: break;
    }
    int s = aa_index(c);
    if (s < 0) return false;
    cnt[s] = 1.f;
    return true;
}

// Index of a letter's symbol when it is unambiguous, else -1.
static int sym_index(char c, int alpha) {
    float cnt[MSA_MAX_SYMS];
    if (!sym_counts(c, cnt, alpha)) return -1;
    for (int i = 0; i < alpha; ++i) if (cnt[i] == 1.f) return i;
    return -1;
}

// BLOSUM62, rows/cols in ARNDCQEGHILKMFPSTWYV order.
static const float BLOSUM62[20][20] = {
 { 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0},
 {-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3},
 {-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3},
 {-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3},
 { 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1},
 {-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2},
 {-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2},
 { 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3},
 {-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3},
 {-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3},
 {-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1},
 {-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2},
 {-1,-1,-2,-3,-1, 0,-2,-3,-1, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1},
 {-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1},
 {-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2},
 { 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2},
 { 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0},
 {-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3},
 {-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1},
 { 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4},
};

Params protein_params() {
    Params P;
    P.alpha = 20;
    P.kmer_k = 2;                  // ClustalW protein ktuple convention
    P.gap_open = 11.0f;            // ClustalW protein defaults; a 40-family
    P.gap_extend = 1.0f;           // BAliBASE subset sweep found a flat
    P.free_end_gaps = false;       // optimum here (protein MSA is global)
    for (int a = 0; a < 20; ++a)
        for (int b = 0; b < 20; ++b)
            P.sub[a * 20 + b] = BLOSUM62[a][b];
    return P;
}

// ------------------------------------------------------------- codons
// Codon index: base-4 with T=0,C=1,A=2,G=3 -> idx = b0*16 + b1*4 + b2.
// Index 64 = "other": partial (<3 nt) or ambiguous codon, scores 0 vs all.
static int nt4(char c) {
    switch (c) {
        case 'T': case 't': case 'U': case 'u': return 0;
        case 'C': case 'c': return 1;
        case 'A': case 'a': return 2;
        case 'G': case 'g': return 3;
        default: return -1;
    }
}
static const char NT4B[4] = {'T', 'C', 'A', 'G'};

// Codon -> amino-acid index in BLOSUM order (ARNDCQEGHILKMFPSTWYV),
// -1 = stop. Standard genetic code (NCBI gc=1).
static const signed char CODON_AA_STD[64] = {
 // TTT  TTC  TTA  TTG  TCT  TCC  TCA  TCG  TAT  TAC  TAA  TAG  TGT  TGC  TGA  TGG
    13,  13,  10,  10,  15,  15,  15,  15,  18,  18,  -1,  -1,   4,   4,  -1,  17,
 // CTT  CTC  CTA  CTG  CCT  CCC  CCA  CCG  CAT  CAC  CAA  CAG  CGT  CGC  CGA  CGG
    10,  10,  10,  10,  14,  14,  14,  14,   8,   8,   5,   5,   1,   1,   1,   1,
 // ATT  ATC  ATA  ATG  ACT  ACC  ACA  ACG  AAT  AAC  AAA  AAG  AGT  AGC  AGA  AGG
     9,   9,   9,  12,  16,  16,  16,  16,   2,   2,  11,  11,  15,  15,   1,   1,
 // GTT  GTC  GTA  GTG  GCT  GCC  GCA  GCG  GAT  GAC  GAA  GAG  GGT  GGC  GGA  GGG
    19,  19,  19,  19,   0,   0,   0,   0,   3,   3,   6,   6,   7,   7,   7,   7,
};

// Vertebrate mitochondrial (NCBI gc=2): TGA->W, ATA->M, AGA/AGG->stop.
static const signed char CODON_AA_MT[64] = {
    13,  13,  10,  10,  15,  15,  15,  15,  18,  18,  -1,  -1,   4,   4,  17,  17,
    10,  10,  10,  10,  14,  14,  14,  14,   8,   8,   5,   5,   1,   1,   1,   1,
     9,   9,  12,  12,  16,  16,  16,  16,   2,   2,  11,  11,  15,  15,  -1,  -1,
    19,  19,  19,  19,   0,   0,   0,   0,   3,   3,   6,   6,   7,   7,   7,   7,
};

static const signed char* codon_aa_table(int gc_def) {
    return gc_def == 2 ? CODON_AA_MT : CODON_AA_STD;
}

Params codon_params(int gc_def) {
    Params P;
    P.alpha = 65;
    P.gc_def = gc_def;
    P.kmer_k = 3;                  // 3 codons = 9 nt
    P.gap_open = 11.0f;            // codon units; protein-scale defaults,
    P.gap_extend = 1.0f;           // swept after first real-gene runs
    P.free_end_gaps = true;        // CDS fragments (CYTB 207 vs 1137 bp)
    const signed char* aa = codon_aa_table(gc_def);
    for (int c1 = 0; c1 < 64; ++c1) {
        int a1 = aa[c1];
        int b0 = c1 >> 4, b1 = (c1 >> 2) & 3, b2 = c1 & 3;
        for (int c2 = 0; c2 < 64; ++c2) {
            int a2 = aa[c2];
            float s;
            if (a1 < 0 || a2 < 0)
                s = (a1 < 0 && a2 < 0) ? 0.f : -P.codon_stop_pen;
            else {
                int same = (b0 == (c2 >> 4)) + (b1 == ((c2 >> 2) & 3))
                         + (b2 == (c2 & 3));
                s = BLOSUM62[a1][a2] + P.codon_nt_bonus * same;
            }
            P.sub[c1 * 65 + c2] = s;
        }
        P.sub[c1 * 65 + 64] = 0.f;   // "other" codon: neutral
    }
    for (int c2 = 0; c2 < 65; ++c2) P.sub[64 * 65 + c2] = 0.f;
    return P;
}

std::vector<std::string> codon_encode(const std::vector<std::string>& seqs,
                                      int gc_def,
                                      std::vector<CodonQc>* qc) {
    const signed char* aa = codon_aa_table(gc_def);
    std::vector<std::string> out(seqs.size());
    if (qc) qc->assign(seqs.size(), {});
    for (size_t s = 0; s < seqs.size(); ++s) {
        const std::string& q = seqs[s];
        // frame = fewest stop codons among the three forward frames
        int best = 0, bestst = INT_MAX;
        for (int f = 0; f < 3; ++f) {
            int st = 0;
            for (size_t i = f; i + 2 < q.size(); i += 3) {
                int a = nt4(q[i]), b = nt4(q[i + 1]), c = nt4(q[i + 2]);
                if (a < 0 || b < 0 || c < 0) continue;
                if (aa[a * 16 + b * 4 + c] < 0) ++st;
            }
            if (st < bestst) { bestst = st; best = f; }
        }
        std::string& e = out[s];
        e.reserve(q.size() / 3 + 1);
        int partial = 0;
        for (size_t i = best; i + 2 < q.size(); i += 3) {
            int a = nt4(q[i]), b = nt4(q[i + 1]), c = nt4(q[i + 2]);
            e.push_back((char)(128 + ((a < 0 || b < 0 || c < 0)
                                   ? 64 : a * 16 + b * 4 + c)));
        }
        if ((q.size() - best) % 3) {
            ++partial;
            e.push_back((char)(128 + 64)); // trailing partial keeps length honest
        }
        if (qc) (*qc)[s] = {best, bestst, partial};
    }
    return out;
}

std::vector<std::string> codon_decode(const std::vector<std::string>& rows) {
    std::vector<std::string> out(rows.size());
    for (size_t s = 0; s < rows.size(); ++s) {
        std::string& o = out[s];
        o.reserve(rows[s].size() * 3);
        for (char ch : rows[s]) {
            int c = (unsigned char)ch;
            if (c == 45 /*'-'*/) { o += "---"; continue; }
            c -= 128;                     // un-offset the codon token
            if (c < 0 || c > 63) { o += "NNN"; continue; }
            o += NT4B[c >> 4]; o += NT4B[(c >> 2) & 3]; o += NT4B[c & 3];
        }
    }
    return out;
}

Profile profile_from_seq(const std::string& seq, int alpha) {
    Profile p;
    p.nseq = 1;
    p.alpha = alpha;
    p.rows.push_back(seq);
    p.ids.push_back(0);          // caller may overwrite with the true index
    p.cols.resize(seq.size());
    p.occ.resize(seq.size());
    for (size_t i = 0; i < seq.size(); ++i) {
        p.cols[i].fill(0.f);
        float cnt[MSA_MAX_SYMS];
        if (sym_counts(seq[i], cnt, alpha)) {
            for (int b = 0; b < alpha; ++b) p.cols[i][b] = cnt[b];
            p.occ[i] = 1.f;
        } else {
            p.cols[i][alpha] = 1.f;
            p.occ[i] = 0.f;
        }
    }
    return p;
}

void profile_update_counts(Profile& p) {
    const int al = p.alpha;
    const int L = p.rows.empty() ? 0 : (int)p.rows[0].size();
    p.cols.assign(L, decltype(p.cols)::value_type{});
    p.occ.assign(L, 0.f);
    for (const auto& row : p.rows) {
        assert((int)row.size() == L);
        for (int i = 0; i < L; ++i) {
            float cnt[MSA_MAX_SYMS];
            if (sym_counts(row[i], cnt, al)) {
                for (int b = 0; b < al; ++b) p.cols[i][b] += cnt[b];
                p.occ[i] += 1.f;
            } else {
                p.cols[i][al] += 1.f;
            }
        }
    }
    for (int i = 0; i < L; ++i) {
        for (int b = 0; b <= al; ++b) p.cols[i][b] /= (float)p.nseq;
        p.occ[i] /= (float)p.nseq;
    }
}

// ------------------------------------------------------------ distances
// S_ij = |kmers_i ∩ kmers_j|,  D = 1 - S / min(|k_i|, |k_j|)
// (MAFFT-style fragment correction: a fragment contained in a longer
// sequence is close to it, not distant).
// Rolling base-alpha k-mer hash kept mod alpha^k (for alpha=4 this is
// exactly the old 2-bit shift + mask). k must satisfy alpha^k <= 2^63.
static uint64_t kmer_base(int alpha, int k) {
    uint64_t base = 1;
    for (int i = 0; i < k; ++i) base *= (uint64_t)alpha;
    return base;
}

std::vector<float> kmer_distances(const std::vector<std::string>& seqs,
                                  int k, int alpha) {
    const int n = (int)seqs.size();
    std::vector<std::unordered_map<uint64_t, bool>> sets(n);
    const uint64_t kbase = kmer_base(alpha, k);
    for (int s = 0; s < n; ++s) {
        const std::string& q = seqs[s];
        uint64_t h = 0;
        int run = 0;
        for (size_t i = 0; i < q.size(); ++i) {
            // only unambiguous letters may extend a k-mer
            int b = sym_index(q[i], alpha);
            if (b < 0) { run = 0; h = 0; continue; }
            h = (h * (uint64_t)alpha + (uint64_t)b) % kbase;
            if (++run >= k) sets[s][h] = true;
        }
    }
    std::vector<float> D((size_t)n * (n - 1) / 2);
    auto idx = [n](int i, int j) {
        if (i > j) std::swap(i, j);
        return (size_t)i * n - (size_t)i * (i + 1) / 2 + (j - i - 1);
    };
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const auto& A = sets[i].size() <= sets[j].size() ? sets[i] : sets[j];
            const auto& B = sets[i].size() <= sets[j].size() ? sets[j] : sets[i];
            size_t inter = 0;
            for (const auto& kv : A) if (B.count(kv.first)) ++inter;
            size_t mn = std::min(sets[i].size(), sets[j].size());
            D[idx(i, j)] = mn ? 1.0f - (float)inter / (float)mn : 1.0f;
        }
    }
    return D;
}

// Contiguous-range parallel map: fn(lo,hi) on disjoint [lo,hi) slices whose
// union is [0,total). Results are deterministic iff fn's writes are disjoint
// (true for every caller below).
static void parallel_for(int total, int nthreads,
                         const std::function<void(int, int)>& fn) {
    nthreads = std::min(nthreads, total);
    if (nthreads <= 1) { fn(0, total); return; }
    std::vector<std::thread> ts;
    const int chunk = (total + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; ++t) {
        const int lo = t * chunk, hi = std::min(total, lo + chunk);
        if (lo >= hi) break;
        ts.emplace_back([&fn, lo, hi] { fn(lo, hi); });
    }
    for (auto& th : ts) th.join();
}

// Multithreaded kmer_distances: same outputs, bit-exact (every write is to a
// disjoint slot; no float accumulation is reordered).
std::vector<float> kmer_distances_mt(const std::vector<std::string>& seqs,
                                     int k, int threads, int alpha) {
    const int n = (int)seqs.size();
    std::vector<std::unordered_map<uint64_t, bool>> sets(n);
    const uint64_t kbase = kmer_base(alpha, k);
    parallel_for(n, threads, [&](int lo, int hi) {
        for (int s = lo; s < hi; ++s) {
            const std::string& q = seqs[s];
            uint64_t h = 0;
            int run = 0;
            for (size_t i = 0; i < q.size(); ++i) {
                int b = sym_index(q[i], alpha);
                if (b < 0) { run = 0; h = 0; continue; }
                h = (h * (uint64_t)alpha + (uint64_t)b) % kbase;
                if (++run >= k) sets[s][h] = true;
            }
        }
    });
    std::vector<float> D((size_t)n * (n - 1) / 2);
    auto idx = [n](int i, int j) {
        if (i > j) std::swap(i, j);
        return (size_t)i * n - (size_t)i * (i + 1) / 2 + (j - i - 1);
    };
    parallel_for(n, threads, [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const auto& A = sets[i].size() <= sets[j].size() ? sets[i] : sets[j];
                const auto& B = sets[i].size() <= sets[j].size() ? sets[j] : sets[i];
                size_t inter = 0;
                for (const auto& kv : A) if (B.count(kv.first)) ++inter;
                size_t mn = std::min(sets[i].size(), sets[j].size());
                D[idx(i, j)] = mn ? 1.0f - (float)inter / (float)mn : 1.0f;
            }
        }
    });
    return D;
}

// -------------------------------------------------------------------- NJ
// Deterministic neighbor joining over a dense matrix sized for all nodes
// (leaves 0..n-1, internal nodes n..2n-2, root = 2n-2).
// Ties break on smallest (i,j) index pair.
Tree nj_tree(const std::vector<float>& Dp, int n) {
    Tree t;
    if (n == 1) { t.nodes.resize(1); t.root = 0; return t; }
    const int NN = 2 * n - 1;
    auto idx = [n](int i, int j) {
        if (i > j) std::swap(i, j);
        return (size_t)i * n - (size_t)i * (i + 1) / 2 + (j - i - 1);
    };
    std::vector<std::vector<double>> d(NN, std::vector<double>(NN, 0));
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            d[i][j] = d[j][i] = Dp[idx(i, j)];

    t.nodes.resize(n);
    std::vector<int> alive(n);
    for (int i = 0; i < n; ++i) alive[i] = i;

    while (alive.size() > 2) {
        int m = (int)alive.size();
        std::vector<double> r(m, 0);
        for (int a = 0; a < m; ++a)
            for (int b = 0; b < m; ++b) if (a != b)
                r[a] += d[alive[a]][alive[b]];
        double best = 0; int bi = -1, bj = -1;
        for (int a = 0; a < m; ++a)
            for (int b = a + 1; b < m; ++b) {
                double q = (m - 2) * d[alive[a]][alive[b]] - r[a] - r[b];
                if (bi < 0 || q < best) { best = q; bi = a; bj = b; }
            }
        int xi = alive[bi], xj = alive[bj];
        int u = (int)t.nodes.size();
        t.nodes.push_back({xi, xj});
        std::vector<int> rest;
        for (int a = 0; a < m; ++a) if (a != bi && a != bj) rest.push_back(alive[a]);
        alive = rest;
        for (int v : rest) {
            d[u][v] = d[v][u] = 0.5 * (d[xi][v] + d[xj][v] - d[xi][xj]);
        }
        alive.push_back(u);
    }
    t.root = (int)t.nodes.size();
    t.nodes.push_back({alive[0], alive[1]});
    return t;
}

// Multithreaded nj_tree: bit-exact with the sequential version. Two rules
// keep it so: (1) each r[a] accumulates over b in the same sequential order
// -- only the ROWS are split across threads, no float sum is reordered;
// (2) the Q-matrix argmin is a per-range minimum merged in (a,b) scan order,
// which reproduces the sequential first-minimum-wins tie break exactly.
Tree nj_tree_mt(const std::vector<float>& Dp, int n, int threads) {
    Tree t;
    if (n == 1) { t.nodes.resize(1); t.root = 0; return t; }
    const int NN = 2 * n - 1;
    auto idx = [n](int i, int j) {
        if (i > j) std::swap(i, j);
        return (size_t)i * n - (size_t)i * (i + 1) / 2 + (j - i - 1);
    };
    std::vector<std::vector<double>> d(NN, std::vector<double>(NN, 0));
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            d[i][j] = d[j][i] = Dp[idx(i, j)];

    t.nodes.resize(n);
    std::vector<int> alive(n);
    for (int i = 0; i < n; ++i) alive[i] = i;

    while (alive.size() > 2) {
        int m = (int)alive.size();
        std::vector<double> r(m, 0);
        parallel_for(m, threads, [&](int lo, int hi) {
            for (int a = lo; a < hi; ++a)
                for (int b = 0; b < m; ++b) if (a != b)
                    r[a] += d[alive[a]][alive[b]];
        });
        // per-range argmin, merged in scan order
        const int nt = std::max(1, std::min(threads, m));
        std::vector<double> tbest(nt, 0);
        std::vector<int>    tbi(nt, -1), tbj(nt, -1);
        {
            const int chunk = (m + nt - 1) / nt;
            std::vector<std::thread> ts;
            for (int t = 0; t < nt; ++t) {
                const int lo = t * chunk, hi = std::min(m, lo + chunk);
                if (lo >= hi) break;
                ts.emplace_back([&, t, lo, hi] {
                    double best = 0; int bi = -1, bj = -1;
                    for (int a = lo; a < hi; ++a)
                        for (int b = a + 1; b < m; ++b) {
                            double q = (m - 2) * d[alive[a]][alive[b]] - r[a] - r[b];
                            if (bi < 0 || q < best) { best = q; bi = a; bj = b; }
                        }
                    tbest[t] = best; tbi[t] = bi; tbj[t] = bj;
                });
            }
            for (auto& th : ts) th.join();
        }
        double best = 0; int bi = -1, bj = -1;
        for (int t = 0; t < nt; ++t) {
            if (tbi[t] < 0) continue;
            if (bi < 0 || tbest[t] < best) { best = tbest[t]; bi = tbi[t]; bj = tbj[t]; }
        }
        int xi = alive[bi], xj = alive[bj];
        int u = (int)t.nodes.size();
        t.nodes.push_back({xi, xj});
        std::vector<int> rest;
        for (int a = 0; a < m; ++a) if (a != bi && a != bj) rest.push_back(alive[a]);
        alive = rest;
        for (int v : rest) {
            d[u][v] = d[v][u] = 0.5 * (d[xi][v] + d[xj][v] - d[xi][xj]);
        }
        alive.push_back(u);
    }
    t.root = (int)t.nodes.size();
    t.nodes.push_back({alive[0], alive[1]});
    return t;
}

// ------------------------------------------------------------- DP scoring
static float sub_score(int a, int b, const Params& P) {
    if (a == b) return P.match;
    bool ts = (a == 0 && b == 2) || (a == 2 && b == 0) ||   // A<->G
              (a == 1 && b == 3) || (a == 3 && b == 1);    // C<->T
    return ts ? P.ts : P.tv;
}

// Column-vs-column weighted sum-of-pairs; the gap frequency (slot 4)
// never participates.
static float col_score(const Profile& A, int i, const Profile& B, int j,
                       const Params& P) {
    const int al = P.alpha;
    float s = 0;
    for (int a = 0; a < al; ++a) {
        float fa = A.cols[i][a];
        if (fa == 0) continue;
        for (int b = 0; b < al; ++b) {
            float fb = B.cols[j][b];
            if (fb == 0) continue;
            s += fa * fb * (al == 4 ? sub_score(a, b, P)
                                    : P.sub[a * al + b]);
        }
    }
    return s;
}

// ------------------------------------------------------- semiglobal DP
// Gotoh 3-state profile-vs-profile.
//   M[i][j]  : columns A_i-1, B_j-1 paired
//   Ix[i][j] : column A_i-1 aligned opposite a gap in B (consumes A)
//   Iy[i][j] : column B_j-1 aligned opposite a gap in A (consumes B)
// Gap-open is scaled by the occupancy of the column the new gap lands
// OPPOSITE (ClustalW/MUSCLE gap-attraction): a gap opposite column X_i
// costs  gap_open * occ_X_i  to open, gap_extend to extend.
// free_end_gaps: first row/col = 0 and the best cell on the last row or
// column is the endpoint -- columns outside the span are emitted as
// prefix/suffix blocks by the merge (semiglobal, fragment-friendly).
AlignResult align_profiles(const Profile& A, const Profile& B,
                           const Params& P) {
    const int M = A.ncols(), N = B.ncols();
    const float NEG = -1e30f;
    std::vector<float> mM((size_t)(M + 1) * (N + 1), NEG),
                       mIx((size_t)(M + 1) * (N + 1), NEG),
                       mIy((size_t)(M + 1) * (N + 1), NEG);
    auto at = [N](int i, int j) { return (size_t)i * (N + 1) + j; };

    mM[0] = 0;
    for (int i = 1; i <= M; ++i) {
        // end-gap run: open at the last consumed column's position-specific
        // price; extensions flat (TWILIGHT gapEnds defaults to gapExtend)
        mIx[at(i, 0)] = P.free_end_gaps ? 0 : -(psgp_open(A.occ[i - 1], P) + (i - 1) * P.gap_extend);
        mM[at(i, 0)]  = P.free_end_gaps ? 0 : NEG;
    }
    for (int j = 1; j <= N; ++j) {
        mIy[at(0, j)] = P.free_end_gaps ? 0 : -(psgp_open(B.occ[j - 1], P) + (j - 1) * P.gap_extend);
        mM[at(0, j)]  = P.free_end_gaps ? 0 : NEG;
    }

    // Decisions are recorded per cell, not re-derived by float comparison at
    // traceback time: (s+mx)-s is not guaranteed to equal mx in floating
    // point, and a recomputed predecessor that matches NONE of the three
    // states would silently fall through to a default op. The kernel records
    // the same byte layout (hsrc|xext|yext); this IS the spec.
    std::vector<uint8_t> dir((size_t)(M + 1) * (N + 1), 0);
    auto dt = [N](int i, int j) { return (size_t)i * (N + 1) + j; };
    constexpr uint8_t H_M = 0, H_IX = 1, H_IY = 2, X_EXT = 4, Y_EXT = 8;

    for (int i = 1; i <= M; ++i) {
        for (int j = 1; j <= N; ++j) {
            float s = col_score(A, i - 1, B, j - 1, P);
            // gap in B opposite A_{i-1}: position-specific penalty
            float openB = psgp_open(A.occ[i - 1], P);
            float extB  = psgp_ext (A.occ[i - 1], P);
            // gap in A opposite B_{j-1}
            float openA = psgp_open(B.occ[j - 1], P);
            float extA  = psgp_ext (B.occ[j - 1], P);
            float oIx = mM[at(i - 1, j)] - openB;
            float eIx = mIx[at(i - 1, j)] - extB;
            float oIy = mM[at(i, j - 1)] - openA;
            float eIy = mIy[at(i, j - 1)] - extA;
            mIx[at(i, j)] = std::max(oIx, eIx);
            mIy[at(i, j)] = std::max(oIy, eIy);
            float mx = mM[at(i - 1, j - 1)]; uint8_t h = H_M;
            if (mIx[at(i - 1, j - 1)] > mx) { mx = mIx[at(i - 1, j - 1)]; h = H_IX; }
            if (mIy[at(i - 1, j - 1)] > mx) { mx = mIy[at(i - 1, j - 1)]; h = H_IY; }
            mM[at(i, j)] = s + mx;
            dir[dt(i, j)] = h | (eIx > oIx ? X_EXT : 0) | (eIy > oIy ? Y_EXT : 0);
        }
    }

    int ei = M, ej = N; float best = mM[at(M, N)]; char estate = 'M';
    if (P.free_end_gaps) {
        best = NEG;
        for (int i = 0; i <= M; ++i) {
            if (mM[at(i, N)] > best) { best = mM[at(i, N)]; ei = i; ej = N; estate = 'M'; }
            if (mIx[at(i, N)] > best) { best = mIx[at(i, N)]; ei = i; ej = N; estate = 'I'; }
        }
        for (int j = 0; j <= N; ++j) {
            if (mM[at(M, j)] > best) { best = mM[at(M, j)]; ei = M; ej = j; estate = 'M'; }
            if (mIy[at(M, j)] > best) { best = mIy[at(M, j)]; ei = M; ej = j; estate = 'D'; }
        }
    }

    std::string cig;
    int i = ei, j = ej; char st = estate;
    while (i > 0 || j > 0) {
        if (st == 'M') {
            if (i == 0 || j == 0) break;
            cig.push_back('M');
            uint8_t h = dir[dt(i, j)] & 3;
            --i; --j;
            st = (h == H_M) ? 'M' : (h == H_IX ? 'I' : 'D');
        } else if (st == 'I') {
            if (i == 0) break;
            cig.push_back('I');
            if (j > 0) st = (dir[dt(i, j)] & X_EXT) ? 'I' : 'M';
            --i;
        } else { // 'D'
            if (j == 0) break;
            cig.push_back('D');
            if (i > 0) st = (dir[dt(i, j)] & Y_EXT) ? 'D' : 'M';
            --j;
        }
    }
    std::reverse(cig.begin(), cig.end());
    AlignResult r;
    r.score = best;
    r.cigar = cig;
    r.ai = i; r.aj = j;
    r.bi = ei; r.bj = ej;
    return r;
}

// -------------------------------------------------------------- merging
// Output layout (deterministic):
//   [A-prefix block][B-prefix block][cigar span][A-suffix][B-suffix]
// A-prefix = columns of A before the aligned span (B emits gaps), and
// symmetrically. Cigar ops: M consumes both, I consumes A (B gap),
// D consumes B (A gap).
Profile merge_profiles(const Profile& A, const Profile& B,
                       const AlignResult& aln) {
    Profile out;
    out.nseq = A.nseq + B.nseq;
    out.alpha = A.alpha;
    const int w = aln.ai + aln.aj + (int)aln.cigar.size()
                  + (A.ncols() - aln.bi) + (B.ncols() - aln.bj);

    for (const auto& row : A.rows) {
        std::string s; s.reserve(w);
        int cA = aln.ai, cB = aln.aj;
        for (int c = 0; c < aln.ai; ++c) s.push_back(row[c]);
        s.append(aln.aj, '-');
        for (char op : aln.cigar) {
            if (op == 'M')      { s.push_back(row[cA++]); ++cB; }
            else if (op == 'I') { s.push_back(row[cA++]); }
            else                { s.push_back('-');       ++cB; }
        }
        for (; cA < A.ncols(); ++cA) s.push_back(row[cA]);
        s.append(B.ncols() - aln.bj, '-');
        while ((int)s.size() < w) s.push_back('-');
        out.rows.push_back(s);
    }
    for (const auto& row : B.rows) {
        std::string s; s.reserve(w);
        int cA = aln.ai, cB = aln.aj;
        s.append(aln.ai, '-');
        for (int c = 0; c < aln.aj; ++c) s.push_back(row[c]);
        for (char op : aln.cigar) {
            if (op == 'M')      { s.push_back(row[cB++]); ++cA; }
            else if (op == 'I') { s.push_back('-');       ++cA; }
            else                { s.push_back(row[cB++]); }
        }
        s.append(A.ncols() - aln.bi, '-');
        for (; cB < B.ncols(); ++cB) s.push_back(row[cB]);
        while ((int)s.size() < w) s.push_back('-');
        out.rows.push_back(s);
    }
    out.ids = A.ids;
    out.ids.insert(out.ids.end(), B.ids.begin(), B.ids.end());
    for (auto& s : out.rows) assert((int)s.size() == w);
    profile_update_counts(out);
    return out;
}

// -------------------------------------------------- gappy-column heuristic
// TWILIGHT --remove-gappy equivalent: contiguous runs of columns whose gap
// fraction exceeds the threshold are removed before the DP and re-inserted
// into the CIGAR afterwards (see cigar_expand_gappy).
GappyStrip profile_strip(const Profile& p, float thr) {
    GappyStrip s;
    const int n = p.ncols();
    for (int c = 0; c < n;) {
        if (p.cols[c][p.alpha] > thr) {         // gappy column -> run
            int len = 0;
            while (c + len < n && p.cols[c + len][p.alpha] > thr) ++len;
            s.run_pos.push_back((int)s.prof.cols.size()); // anchor in reduced
            s.run_start.push_back(c);
            s.run_len.push_back(len);
            c += len;
        } else {
            s.prof.cols.push_back(p.cols[c]);
            s.prof.occ.push_back(p.occ[c]);
            ++c;
        }
    }
    // rows keep the ORIGINAL columns (needed for reinsertion / merge)
    s.prof.rows  = p.rows;
    s.prof.ids   = p.ids;
    s.prof.nseq  = p.nseq;
    s.prof.alpha = p.alpha;
    s.orig_cols = n;
    return s;
}

namespace {
// Build a profile holding only ORIGINAL columns [start, start+len) of `p`.
// `p` may be a reduced (stripped) profile whose cols/occ no longer span the
// original coordinates -- but its rows always do. Slicing reduced cols by
// original indices would read out of bounds; recounting from the row
// substrings reproduces the original column data exactly.
Profile sub_profile(const Profile& p, int start, int len) {
    Profile q;
    q.rows.reserve(p.rows.size());
    for (const auto& r : p.rows) q.rows.push_back(r.substr(start, len));
    q.ids = p.ids; q.nseq = p.nseq; q.alpha = p.alpha;
    profile_update_counts(q);
    return q;
}
} // namespace

// Reinsert stripped runs into a reduced-coordinate CIGAR. The walk counts
// REDUCED columns consumed (i for A, j for B); a run anchored at pos sits
// before the pos-th kept column. Semiglobal skipped ends are emitted as
// plain I/D ops so the result fully consumes both originals (ai=aj=0).
AlignResult cigar_expand_gappy(const AlignResult& aln,
                               const GappyStrip& sa,
                               const GappyStrip& sb,
                               const Params& P) {
    AlignResult r;
    r.score = aln.score;
    int i = 0, j = 0;                    // reduced columns consumed
    size_t ra = 0, rb = 0;               // run cursors
    const int Ka = sa.prof.ncols(), Kb = sb.prof.ncols();

    Params g = P; g.free_end_gaps = false;   // coincident runs: global mini
    // flush runs anchored at (i,j); when both sides coincide, mini-align
    // their column blocks against each other
    auto flush = [&] {
        for (;;) {
            bool ha = ra < sa.run_pos.size() && sa.run_pos[ra] == i;
            bool hb = rb < sb.run_pos.size() && sb.run_pos[rb] == j;
            if (!ha && !hb) return;
            if (ha && hb) {
                Profile qa = sub_profile(sa.prof, sa.run_start[ra], sa.run_len[ra]);
                Profile qb = sub_profile(sb.prof, sb.run_start[rb], sb.run_len[rb]);
                AlignResult mini = align_profiles(qa, qb, g);
                r.cigar += mini.cigar;
                ++ra; ++rb;
            } else if (ha) {
                r.cigar.append((size_t)sa.run_len[ra], 'I'); ++ra;
            } else {
                r.cigar.append((size_t)sb.run_len[rb], 'D'); ++rb;
            }
        }
    };

    // skipped prefixes (semiglobal): kept cols + anchored runs, as raw ops
    for (; i < aln.ai; ++i) { flush(); r.cigar += 'I'; } flush();
    for (; j < aln.aj; ++j) { flush(); r.cigar += 'D'; } flush();
    for (char op : aln.cigar) {
        flush();
        r.cigar += op;
        if (op == 'M') { ++i; ++j; }
        else if (op == 'I') ++i; else ++j;
    }
    // suffixes
    for (; i < Ka; ++i) { flush(); r.cigar += 'I'; } flush();
    for (; j < Kb; ++j) { flush(); r.cigar += 'D'; } flush();
    // The cigar now consumes every ORIGINAL column of both profiles, so the
    // merge span covers the whole profiles: ai=aj=0, bi/bj = orig widths.
    r.ai = 0; r.aj = 0;
    r.bi = sa.orig_cols;
    r.bj = sb.orig_cols;
    return r;
}

// --------------------------------------------------------------- levels
std::vector<std::vector<int>> tree_levels(const Tree& t) {
    std::vector<int> lvl(t.nodes.size(), 0);
    std::vector<std::vector<int>> out;
    // nodes were appended in join order (n, n+1, ..., root): parents always
    // come after their children, so a single ascending pass assigns levels.
    for (size_t u = 0; u < t.nodes.size(); ++u) {
        const Node& nd = t.nodes[u];
        if (nd.left < 0) continue;
        lvl[u] = std::max(lvl[nd.left], lvl[nd.right]) + 1;
        if ((int)out.size() < lvl[u]) out.resize(lvl[u]);
        out[lvl[u] - 1].push_back((int)u);
    }
    return out;
}

// --------------------------------------------------------------- driver
static void postorder(const Tree& t, int u, std::vector<int>& out) {
    const Node& nd = t.nodes[u];
    if (nd.left < 0) { out.push_back(u); return; }
    postorder(t, nd.left, out);
    postorder(t, nd.right, out);
    out.push_back(u);
}

std::vector<std::string> msa_align_with_tree(
        const std::vector<std::string>& seqs, const Tree& tree,
        const Params& P) {
    const int n = (int)seqs.size();
    std::vector<Profile> profs(tree.nodes.size());
    for (int i = 0; i < n; ++i) {
        profs[i] = profile_from_seq(seqs[i], P.alpha);
        profs[i].ids[0] = i;
    }
    std::vector<int> order;
    postorder(tree, tree.root, order);
    for (int u : order) {
        const Node& nd = tree.nodes[u];
        if (nd.left < 0) continue;
        AlignResult aln;
        if (P.gappy > 0.0f) {
            GappyStrip sa = profile_strip(profs[nd.left],  P.gappy);
            GappyStrip sb = profile_strip(profs[nd.right], P.gappy);
            aln = cigar_expand_gappy(
                align_profiles(sa.prof, sb.prof, P), sa, sb, P);
        } else {
            aln = align_profiles(profs[nd.left], profs[nd.right], P);
        }
        profs[u] = merge_profiles(profs[nd.left], profs[nd.right], aln);
    }
    // return rows in INPUT order (merge order follows the guide tree)
    std::vector<std::string> out(n);
    const Profile& root = profs[tree.root];
    for (int k = 0; k < n; ++k) out[root.ids[k]] = root.rows[k];
    return out;
}

std::vector<std::string> msa_align(const std::vector<std::string>& seqs,
                                   const Params& P, Tree* guide_out) {
    std::vector<float> D = kmer_distances(seqs, P.kmer_k, P.alpha);
    Tree t = nj_tree(D, (int)seqs.size());
    if (guide_out) *guide_out = t;
    return msa_align_with_tree(seqs, t, P);
}

std::string tree_to_newick(const Tree& t,
                           const std::vector<std::string>& names) {
    std::string s;
    std::function<void(int)> emit = [&](int u) {
        const Node& nd = t.nodes[u];
        if (nd.left < 0) {                      // leaf: id is the seq index
            const std::string& nm = names[u];
            bool quote = nm.find_first_of(" \t()[]':;,") != std::string::npos;
            s += quote ? "'" + nm + "'" : nm;
            return;
        }
        s += '(';
        emit(nd.left);
        s += ',';
        emit(nd.right);
        s += ')';
    };
    emit(t.root);
    s += ";\n";
    return s;
}

} // namespace genomsa
