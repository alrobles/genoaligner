// CPU reference implementation of the genoaligner MSA engine.
// Semantics are fixed here BEFORE the GPU kernel exists -- every later
// kernel is validated against this file's outputs, not the other way.
#include <genoaligner/msa/msa.hpp>
#include <unordered_map>
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

Profile profile_from_seq(const std::string& seq) {
    Profile p;
    p.nseq = 1;
    p.rows.push_back(seq);
    p.ids.push_back(0);          // caller may overwrite with the true index
    p.cols.resize(seq.size());
    p.occ.resize(seq.size());
    for (size_t i = 0; i < seq.size(); ++i) {
        float cnt[4];
        if (iupac_counts(seq[i], cnt)) {
            for (int b = 0; b < 4; ++b) p.cols[i][b] = cnt[b];
            p.cols[i][4] = 0.f;
            p.occ[i] = 1.f;
        } else {
            for (int b = 0; b < 4; ++b) p.cols[i][b] = 0.f;
            p.cols[i][4] = 1.f;
            p.occ[i] = 0.f;
        }
    }
    return p;
}

void profile_update_counts(Profile& p) {
    const int L = p.rows.empty() ? 0 : (int)p.rows[0].size();
    p.cols.assign(L, {0.f, 0.f, 0.f, 0.f, 0.f});
    p.occ.assign(L, 0.f);
    for (const auto& row : p.rows) {
        assert((int)row.size() == L);
        for (int i = 0; i < L; ++i) {
            float cnt[4];
            if (iupac_counts(row[i], cnt)) {
                for (int b = 0; b < 4; ++b) p.cols[i][b] += cnt[b];
                p.occ[i] += 1.f;
            } else {
                p.cols[i][4] += 1.f;
            }
        }
    }
    for (int i = 0; i < L; ++i) {
        for (int b = 0; b < 5; ++b) p.cols[i][b] /= (float)p.nseq;
        p.occ[i] /= (float)p.nseq;
    }
}

// ------------------------------------------------------------ distances
// S_ij = |kmers_i ∩ kmers_j|,  D = 1 - S / min(|k_i|, |k_j|)
// (MAFFT-style fragment correction: a fragment contained in a longer
// sequence is close to it, not distant).
std::vector<float> kmer_distances(const std::vector<std::string>& seqs, int k) {
    const int n = (int)seqs.size();
    std::vector<std::unordered_map<uint64_t, bool>> sets(n);
    const uint64_t kmask = (k >= 31) ? ~0ull : ((1ull << (2 * k)) - 1);
    for (int s = 0; s < n; ++s) {
        const std::string& q = seqs[s];
        uint64_t h = 0;
        int run = 0;
        for (size_t i = 0; i < q.size(); ++i) {
            float cnt[4];
            // only unambiguous ACGT may extend a k-mer
            bool clean = iupac_counts(q[i], cnt) &&
                         (cnt[0] == 1.f || cnt[1] == 1.f ||
                          cnt[2] == 1.f || cnt[3] == 1.f);
            if (!clean) { run = 0; h = 0; continue; }
            int b = cnt[0] == 1.f ? 0 : cnt[1] == 1.f ? 1 : cnt[2] == 1.f ? 2 : 3;
            h = (h << 2) | (uint64_t)b;
            if (++run >= k) sets[s][h & kmask] = true;
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
                                     int k, int threads) {
    const int n = (int)seqs.size();
    std::vector<std::unordered_map<uint64_t, bool>> sets(n);
    const uint64_t kmask = (k >= 31) ? ~0ull : ((1ull << (2 * k)) - 1);
    parallel_for(n, threads, [&](int lo, int hi) {
        for (int s = lo; s < hi; ++s) {
            const std::string& q = seqs[s];
            uint64_t h = 0;
            int run = 0;
            for (size_t i = 0; i < q.size(); ++i) {
                float cnt[4];
                bool clean = iupac_counts(q[i], cnt) &&
                             (cnt[0] == 1.f || cnt[1] == 1.f ||
                              cnt[2] == 1.f || cnt[3] == 1.f);
                if (!clean) { run = 0; h = 0; continue; }
                int b = cnt[0] == 1.f ? 0 : cnt[1] == 1.f ? 1 : cnt[2] == 1.f ? 2 : 3;
                h = (h << 2) | (uint64_t)b;
                if (++run >= k) sets[s][h & kmask] = true;
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
    float s = 0;
    for (int a = 0; a < 4; ++a) {
        float fa = A.cols[i][a];
        if (fa == 0) continue;
        for (int b = 0; b < 4; ++b) {
            float fb = B.cols[j][b];
            if (fb == 0) continue;
            s += fa * fb * sub_score(a, b, P);
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
        mIx[at(i, 0)] = P.free_end_gaps ? 0 : -(P.gap_open * A.occ[i - 1] + (i - 1) * P.gap_extend);
        mM[at(i, 0)]  = P.free_end_gaps ? 0 : NEG;
    }
    for (int j = 1; j <= N; ++j) {
        mIy[at(0, j)] = P.free_end_gaps ? 0 : -(P.gap_open * B.occ[j - 1] + (j - 1) * P.gap_extend);
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
            // gap in B opposite A_{i-1}: scaled by A's occupancy
            float openB = P.gap_open * A.occ[i - 1];
            // gap in A opposite B_{j-1}: scaled by B's occupancy
            float openA = P.gap_open * B.occ[j - 1];
            float oIx = mM[at(i - 1, j)] - openB;
            float eIx = mIx[at(i - 1, j)] - P.gap_extend;
            float oIy = mM[at(i, j - 1)] - openA;
            float eIy = mIy[at(i, j - 1)] - P.gap_extend;
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
        profs[i] = profile_from_seq(seqs[i]);
        profs[i].ids[0] = i;
    }
    std::vector<int> order;
    postorder(tree, tree.root, order);
    for (int u : order) {
        const Node& nd = tree.nodes[u];
        if (nd.left < 0) continue;
        AlignResult aln = align_profiles(profs[nd.left], profs[nd.right], P);
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
    std::vector<float> D = kmer_distances(seqs, P.kmer_k);
    Tree t = nj_tree(D, (int)seqs.size());
    if (guide_out) *guide_out = t;
    return msa_align_with_tree(seqs, t, P);
}

} // namespace genomsa
