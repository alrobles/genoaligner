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
// Maximum alphabet width supported: 20 amino acids (+1 gap slot).
// DNA mode uses symbols 0..3 (A,C,G,T); the gap fraction always lives at
// index `alpha` so indexing is alphabet-agnostic.
constexpr int MSA_MAX_SYMS = 20;

struct Params {
    int    kmer_k       = 5;
    int    alpha        = 4;      // 4 = DNA (IUPAC), 20 = protein
    float  match        = 2.0f;   // S(a,a)              (alpha==4 only)
    float  ts           = -1.0f;  // transition mismatch (alpha==4 only)
    float  tv           = -2.0f;  // transversion        (alpha==4 only)
    // Substitution matrix, row-major alpha*alpha, used when alpha>4.
    // protein_params() fills it with BLOSUM62.
    std::array<float, MSA_MAX_SYMS * MSA_MAX_SYMS> sub{};
    float  gap_open     = 3.0f;   // scaled by opposing occupancy
    float  gap_extend   = 1.0f;
    bool   free_end_gaps = true;  // semiglobal ends (fragments)
    // Position-specific gap penalties (ClustalW/TWILIGHT convention):
    // a column that already contains gaps accepts a new gap more cheaply.
    // gapOpen[c] = occ==1 ? go : max(psgp_min_open*go, psgp_scale*go*occ)
    // gapEx[c]   = occ==1 ? ge : max(psgp_min_ext *ge,           ge*occ)
    // ON by default: +4..7 SIM-SPS on the simulated-truth benchmark.
    bool   psgp         = true;
    float  psgp_scale   = 0.5f;   // nucleotide scale (TWILIGHT uses 0.5)
    float  psgp_min_open = 0.1f;  // floor: fraction of gap_open
    float  psgp_min_ext  = 0.2f;  // floor: fraction of gap_extend
    // Gappy-column heuristic (TWILIGHT --remove-gappy): columns with gap
    // fraction > gappy are stripped before the DP and re-inserted as
    // insertion blocks; runs removed from BOTH profiles at the same
    // position are mini-aligned to each other. 0 disables. Default 0.95.
    float  gappy        = 0.95f;
};

// Position-specific gap penalties -- THE SPEC. The kernel implements the
// identical formula from the occupancy array; keep the two in sync.
inline float psgp_open(float occ, const Params& P) {
    if (!P.psgp || occ >= 1.0f) return P.gap_open * occ;
    const float p = P.psgp_scale * P.gap_open * occ;
    const float fl = P.psgp_min_open * P.gap_open;
    return p > fl ? p : fl;
}
inline float psgp_ext(float occ, const Params& P) {
    if (!P.psgp || occ >= 1.0f) return P.gap_extend;
    const float p = P.gap_extend * occ;
    const float fl = P.psgp_min_ext * P.gap_extend;
    return p > fl ? p : fl;
}

// Protein preset: alpha=20, BLOSUM62 substitution matrix, protein
// distance/DP defaults (2-mer guide tree, ClustalW-scale gap costs).
Params protein_params();

// ---------------------------------------------------------------- profile
struct Profile {
    // Per-column fractional counts. cols[c][s] = fraction of letter s
    // (s in [0,alpha)); cols[c][alpha] = gap fraction.
    // occ[c] = 1 - gap fraction.
    std::vector<std::array<float, MSA_MAX_SYMS + 1>> cols;
    std::vector<float>               occ;
    std::vector<std::string>         rows;  // actual aligned sequences
    std::vector<int>                 ids;   // rows[k] came from input seq ids[k]
    int nseq  = 0;
    int alpha = 4;                  // alphabet width of this profile
    int ncols() const { return (int)cols.size(); }
};

// A single sequence as a one-column-per-symbol profile.
// alpha=4: IUPAC DNA. alpha=20: amino acids (BLOSUM order
// ARNDCQEGHILKMFPSTWYV; B/Z/J map to their two-letter sets, X/O uniform,
// U -> C, '*' and unknown letters count as gap).
Profile profile_from_seq(const std::string& seq, int alpha = 4);
// Recompute fractional column counts + occupancy from `rows`.
void    profile_update_counts(Profile& p);

// ------------------------------------------------------------- distances
// Fragment-corrected k-mer Jaccard distance matrix (lower triangle packed).
std::vector<float> kmer_distances(const std::vector<std::string>& seqs,
                                  int k, int alpha = 4);
// Multithreaded variant: bit-exact same output (disjoint writes only).
std::vector<float> kmer_distances_mt(const std::vector<std::string>& seqs,
                                     int k, int threads, int alpha = 4);

// --------------------------------------------------------------- NJ tree
struct Node { int left = -1, right = -1; };  // children node ids; leaf if <0
struct Tree {
    std::vector<Node> nodes;   // ids 0..n-1 are leaves (seq index), >=n internal
    int root = -1;
};
Tree nj_tree(const std::vector<float>& dist_packed, int n);
// Multithreaded variant: bit-exact same tree (per-row sums keep sequential
// order; the Q argmin merges per-range minima in scan order).
Tree nj_tree_mt(const std::vector<float>& dist_packed, int n, int threads);

// Internal nodes grouped by merge level: level(u) = max(level(children))+1,
// leaves = 0. All nodes in one level are INDEPENDENT -- the batch unit the
// GPU launches one kernel per level over. Levels come back ascending, so
// iterating them in order respects dependencies. Deterministic: within a
// level, nodes keep ascending node-id order.
std::vector<std::vector<int>> tree_levels(const Tree& t);

// Serialize the guide tree as Newick (topology only, no branch lengths).
// names[i] is the label of leaf i (typically the FASTA id).
std::string tree_to_newick(const Tree& t, const std::vector<std::string>& names);

// ------------------------------------------------------- profile-profile
struct AlignResult {
    float score = 0;
    std::string cigar;         // expanded ops over columns: 'M','I'(A only),'D'(B only)
    int   ai = 0, aj = 0;      // aligned span starts (semiglobal)
    int   bi = 0, bj = 0;      // aligned span ends (exclusive)
};
AlignResult align_profiles(const Profile& A, const Profile& B,
                           const Params& P);

// Gappy-column heuristic (Params::gappy > 0, TWILIGHT --remove-gappy).
// profile_strip removes contiguous runs of columns whose gap fraction
// exceeds the threshold; runs records each removed run anchored at the
// reduced-column index it was stripped from (pos in [0, ncols()]).
struct GappyStrip {
    Profile          prof;       // reduced profile (kept columns only)
    std::vector<int> run_pos;    // reduced index each run was removed at
    std::vector<int> run_start;  // original start column of each run
    std::vector<int> run_len;
    int              orig_cols = 0;  // ncols of the profile before stripping
};
GappyStrip profile_strip(const Profile& p, float thr);
// Translate a CIGAR over the reduced profiles back to original-column
// ops: emits a fully-consuming CIGAR (ai=aj=0, bi/bj = full widths) with
// one-sided runs as insertion blocks; coincident runs on both sides are
// mini-aligned globally. Deterministic.
AlignResult cigar_expand_gappy(const AlignResult& aln,
                               const GappyStrip& sa,
                               const GappyStrip& sb,
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

// ------------------------------------------------------------- GPU path
// Level-batched driver: one kernel launch per guide-tree level over its
// independent pairs, reference merge on host. Bit-exact with
// msa_align_with_tree (gated by tests/parity/msa_pipeline_parity.cpp).
// Compiled only under a HIP compiler (src/msa/msa_gpu.cpp).
struct GpuStats {
    double dist_s = 0, tree_s = 0, align_s = 0;
    int    levels = 0, pairs = 0;
    size_t dir_bytes = 0;
};
bool msa_align_gpu(const std::vector<std::string>& seqs, const Params& P,
                   std::vector<std::string>& out, std::string& err,
                   GpuStats* stats = nullptr, Tree* guide_out = nullptr);

} // namespace genomsa
