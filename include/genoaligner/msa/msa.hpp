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
// Maximum alphabet width supported: 65 = 64 codons + 1 "other/partial"
// token (codon mode). DNA mode uses symbols 0..3 (A,C,G,T); protein uses
// 0..19; the gap fraction always lives at index `alpha` so indexing is
// alphabet-agnostic.
constexpr int MSA_MAX_SYMS = 65;

struct Params {
    int    kmer_k       = 5;
    int    alpha        = 4;      // 4 = DNA (IUPAC), 20 = protein, 65 = codon
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
    // Codon mode (alpha==65): penalty when exactly one side of a column
    // pair is a stop codon (MACSE-style), and per-identical-nt bonus on
    // top of the BLOSUM62 amino-acid score.
    float  codon_stop_pen = 60.0f;
    float  codon_nt_bonus = 1.0f;
    int    gc_def         = 1;    // NCBI genetic code (2 = vertebrate mito)
    // Stage-2 codon refinement (codon_refine): MACSE-style frameshift
    // events. codon_refine = passes over the MSA (0 disables);
    // codon_fs = cost of an internal 1-2 nt indel (MACSE -fs 30);
    // codon_fs_term = terminal frameshift cost (first/last codon of a
    // sequence, MACSE -fs_term 10).
    int    codon_refine   = 0;
    float  codon_fs       = 30.0f;
    float  codon_fs_term  = 10.0f;
    // Max raw-nt drift of codon boundaries inside a row's frozen stage-1
    // footprint during refinement (the DP's |s - 3k| limit).
    float  codon_refine_band = 40.0f;
    // codon_local_frame: tokenize each sequence with the local-frame DP
    // (codon_encode_local) instead of a single global frame -- keeps
    // tokens in-frame across internal frameshifts.
    int    codon_local_frame = 0;
    // Encode-DP event cost: deliberately higher than codon_fs so that a
    // lone sequencing-error stop can never be dodged (a single-event
    // escape costs codon_fs_enc but saves only codon_stop_pen=60); a real
    // frameshift still wins because its off-frame tail accrues >=2 stops.
    float  codon_fs_enc   = 80.0f;
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

// -------------------------------------------------------- codon mode
// Codon-aware progressive MSA (MACSE-style): sequences are tokenized to
// codons (one byte per codon, stored as 128 + index where index is 0..63
// in NCBI codon order TTT,TTC,...,GGG and 64 = partial/ambiguous codon,
// scores 0 vs all; the +128 offset keeps tokens disjoint from '-'), the
// engine runs with alpha=65 over a 65x65 substitution matrix built from
// BLOSUM62 of the translated amino acids plus:
//   - codon_stop_pen : charged when exactly one side is a stop codon
//   - codon_nt_bonus : added per identical nt position (0..3)
// Indels are then always whole codons -- the output keeps reading frame.
// gc_def: NCBI genetic code id (1 = standard, 2 = vertebrate mito).
Params codon_params(int gc_def = 1);

// Per-sequence frame selection + tokenization. Returns encoded strings
// (one byte per codon) plus QC: chosen frame and stop count per seq.
struct CodonQc { int frame = 0; int stops = 0; int partial = 0; };
std::vector<std::string> codon_encode(const std::vector<std::string>& seqs,
                                      int gc_def,
                                      std::vector<CodonQc>* qc = nullptr);
// Local-frame variant: a stop-avoiding DP partitions each sequence into
// codon blocks (3 nt -> token) and frameshift blocks (1-2 nt -> kept out
// of the token stream; codon_refine restores them as event columns).
// Rows with internal frameshifts keep downstream tokens in-frame; clean
// rows tokenize identically to codon_encode. Enabled via
// Params::codon_local_frame (implies codon_refine >= 1).
std::vector<std::string> codon_encode_local(
        const std::vector<std::string>& seqs, const Params& P,
        std::vector<CodonQc>* qc = nullptr);
// Expand codon-token aligned rows back to nucleotides (token 64 -> NNN).
std::vector<std::string> codon_decode(const std::vector<std::string>& rows);

// Stage-2 codon refinement -- MACSE-style frameshift events. Realigns each
// raw nucleotide sequence against the codon-column profile of the REST of
// the stage-1 MSA with an extended Gotoh whose move set adds 1-2 nt indels
// charged as frameshift events (codon_fs, or codon_fs_term inside the
// first/last codon of the sequence). A fs event becomes a new column 1-2 nt
// wide holding only that row's residues (the "!" of MACSE); an in-frame
// codon insertion becomes a 3-nt column. A sequence may also cover a codon
// column partially (1-2 nt + gaps, scored against the column's nt
// marginals) -- a deletion frameshift. The output is a NT MSA no longer
// guaranteed %3. rounds>1 rebuilds the codon profile from the refined MSA
// each pass (original codon columns only; insertion columns are
// re-derived). Input order preserved; aln_nt must be the codon_decode()
// output of the stage-1 token MSA (all columns width 3, same row order as
// seqs_nt).
std::vector<std::string> codon_refine(const std::vector<std::string>& seqs_nt,
                                      const std::vector<std::string>& aln_nt,
                                      const Params& P, int rounds);

// GPU variant: the per-row phase DPs run as codon_refine_kernel (one
// thread per row); profile build, traceback and merge stay on host.
// Semantics identical to codon_refine (the DP body is shared source).
// Returns false with err set on device/packing failure -- callers should
// fall back to codon_refine.
bool codon_refine_gpu(const std::vector<std::string>& seqs_nt,
                      const std::vector<std::string>& aln_nt,
                      const Params& P, int rounds,
                      std::vector<std::string>& out, std::string& err,
                      float* kernel_s = nullptr);

// ---- internals shared by codon_refine (host) and codon_refine_gpu ------
namespace detail {
// Codon-column profile of a NT MSA (see codon_prof_build): token counts,
// occupancy, per-position nt counts and each row's token per column.
struct CodonProf {
    int C = 0;
    int nseq = 0;
    std::vector<std::array<float, 65>>            ccnt;
    std::vector<float>                            ocnt;
    std::vector<std::array<std::array<float, 4>, 3>> ncnt;
    std::vector<std::vector<int>>                 tok;
};
CodonProf codon_prof_build(const std::vector<std::string>& rows,
                           const std::vector<int>& cofs);
// One row's placement after realignment: fill[j] = its 3 chars at codon
// column j ("---" = absent); ins[b] = nt blocks inserted before column b
// (b == C = trailing).
struct Place {
    std::vector<std::string>              fill;
    std::vector<std::vector<std::string>> ins;
};
// Reconstruct a row's placement from the phase-DP outputs: pi = footprint
// columns, tr = K*ND direction bytes, bes = endpoint s (-2 = dump raw).
Place codon_trace_place(const std::string& seq, const CodonProf& cp,
                        int self, const Params& P,
                        const std::vector<int>& pi,
                        const uint8_t* tr, int tr_stride, int bes);
// Merge per-row placements into a NT MSA; cofs gets the char offset of
// each kept codon column.
std::vector<std::string> codon_refine_merge(const std::vector<Place>& pls,
                                            int C, std::vector<int>* cofs);
}  // namespace detail

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

// NJ0: exact dense NJ on the device (docs/SPEC_NJ_GPU.md). Bit-exact same
// Tree as nj_tree (same sum order, same first-minimum tie break). On any
// device error returns an empty Tree with `err` set -- callers fall back to
// nj_tree_mt; an approximate tree is never produced. Compiled only under a
// HIP compiler or the CPU shim (src/msa/nj_gpu.cpp).
struct NjStats {
    double upload_s = 0, rounds_s = 0, download_s = 0;
    int    rounds = 0;
    int    exact_fallbacks = 0;   // always 0 in NJ0; reserved for NJ1
    size_t dev_bytes = 0;
};
Tree nj_tree_gpu(const std::vector<float>& dist_packed, int n,
                 std::string& err, NjStats* stats = nullptr);

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
    int    tree_gpu = 0;          // 1 if the guide tree came from nj_tree_gpu
    size_t dir_bytes = 0;
};
bool msa_align_gpu(const std::vector<std::string>& seqs, const Params& P,
                   std::vector<std::string>& out, std::string& err,
                   GpuStats* stats = nullptr, Tree* guide_out = nullptr);

} // namespace genomsa
