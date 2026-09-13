// CPU reference implementation of the genoaligner MSA engine ("genomsa").
//
// Pipeline (see docs/MSA_RESEARCH.md for the design rationale):
//   1. fragment-corrected k-mer Jaccard distances  D = 1 - S/min(S_ii,S_jj)
//   2. deterministic neighbor-joining guide tree (or --guidetree-in)
//   3. post-order traversal: leaf profiles -> profile-vs-profile Gotoh
//      semiglobal DP (free terminal gaps) -> deterministic column
//      interleave merge
//
// A profile is a list of alignment columns; each column stores fractional
// letter counts over {A,C,G,T} (IUPAC codes contribute 1/|set| each), the
// gap count, and its occupancy (non-gap fraction). Column-vs-column score
// is the weighted sum-of-pairs  sum_a sum_b f_i(a)*f_j(b)*S(a,b); the gap
// symbol never participates. Gap-open cost is scaled by the occupancy of
// the opposing column (MUSCLE convention).
//
// Determinism is part of the spec: NJ ties break on smaller index,
// children keep the guide tree's left/right order, DP ties prefer
// M > Ix > Iy, and batch/level scheduling preserves node order.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

namespace genomsa {

// ---------------------------------------------------------------- config
struct Params {
    int    kmer_k       = 5;
    float  match        = 2.0f;   // S(a,a)
    float  ts           = -1.0f;  // transition mismatch (A<->G, C<->T)
    float  tv           = -2.0f;  // transversion mismatch
    float  gap_open     = 3.0f;   // scaled by opposing occupancy
    float  gap_extend   = 1.0f;
    bool   free_end_gaps = true;  // semiglobal ends (fragments)
};

// ---------------------------------------------------------------- profile
struct Profile {
    // Per-column fractional counts. cols[c][0..3] = A,C,G,T fractions,
    // cols[c][4] = gap fraction. occ[c] = 1 - gap fraction.
    std::vector<std::array<float,5>> cols;
    std::vector<float>               occ;
    std::vector<std::string>         rows;  // actual aligned sequences
    std::vector<int>                 ids;   // rows[k] came from input seq ids[k]
    int nseq = 0;
    int ncols() const { return (int)cols.size(); }
};

// A single sequence as a one-column-per-base profile.
Profile profile_from_seq(const std::string& seq);
// Recompute fractional column counts + occupancy from `rows`.
void    profile_update_counts(Profile& p);

// ------------------------------------------------------------- distances
// Fragment-corrected k-mer Jaccard distance matrix (lower triangle packed).
std::vector<float> kmer_distances(const std::vector<std::string>& seqs,
                                  int k);

// --------------------------------------------------------------- NJ tree
struct Node { int left = -1, right = -1; };  // children node ids; leaf if <0
struct Tree {
    std::vector<Node> nodes;   // ids 0..n-1 are leaves (seq index), >=n internal
    int root = -1;
};
Tree nj_tree(const std::vector<float>& dist_packed, int n);

// Internal nodes grouped by merge level: level(u) = max(level(children))+1,
// leaves = 0. All nodes in one level are INDEPENDENT -- the batch unit the
// GPU launches one kernel per level over. Levels come back ascending, so
// iterating them in order respects dependencies. Deterministic: within a
// level, nodes keep ascending node-id order.
std::vector<std::vector<int>> tree_levels(const Tree& t);

// ------------------------------------------------------- profile-profile
struct AlignResult {
    float score = 0;
    std::string cigar;         // expanded ops over columns: 'M','I'(A only),'D'(B only)
    int   ai = 0, aj = 0;      // aligned span starts (semiglobal)
    int   bi = 0, bj = 0;      // aligned span ends (exclusive)
};
AlignResult align_profiles(const Profile& A, const Profile& B,
                           const Params& P);

// Merge two profiles given their column alignment: interleave columns,
// padding each side with gap columns where the other consumes a column.
Profile merge_profiles(const Profile& A, const Profile& B,
                       const AlignResult& aln);

// ------------------------------------------------------------- top level
// Full CPU-reference MSA. Returns rows aligned to equal length.
std::vector<std::string> msa_align(const std::vector<std::string>& seqs,
                                   const Params& P,
                                   Tree* guide_out = nullptr);

// Override the guide tree (Newick-free form: post-order list of merges).
std::vector<std::string> msa_align_with_tree(const std::vector<std::string>& seqs,
                                             const Tree& tree,
                                             const Params& P);

} // namespace genomsa
