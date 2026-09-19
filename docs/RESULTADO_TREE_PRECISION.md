# Tree-precision benchmark — genomsa vs MACSE vs threestep vs truth

Status: **170/170 instances complete** (`data/sim_precision/`, job 29721205).
Design: `codon_sim_v2` generates the true alignment AND true tree per
instance; each aligner's output goes through FastTree (-nt -gtr) and is
scored by normalized RF against the true tree. `sim_true.fasta` gets the
same FastTree pass as the estimator noise floor.

## Headline result

| Cell | genomsa | genomsa_lf | MACSE | threestep |
|---|---|---|---|---|
| non-fs cells (12) | ~0 | ~0 | ~0 | ~0* |
| fs-0.1 | 0.010 | 0.003 | 0.006 | 0.236 |
| fs-0.5 | 0.138 | 0.021 | 0.009 | 0.662 |
| fs-1 | 0.281 | 0.058 | 0.021 | 0.882 |
| fs-3 | 0.486 | 0.174 | 0.022 | 1.000 |
| pseudo-mix | 0.194 | 0.046 | 0.004 | 0.982 |

*threestep collapses (RF 0.5-1.0) whenever ORFs break (frag/noisy/stop).

**Noise floor = 0 everywhere**: FastTree recovers the exact true tree
from the true alignment — every RF point above zero is aligner damage.

## Read

1. Aligner choice is irrelevant to the tree unless frameshifts are
   present; then base genomsa is catastrophic (RF 0.14-0.49) and
   --local-frame recovers ~2/3 of the damage MACSE avoids (lf RF 3-8x
   macse but 3-10x better than base).
2. threestep (MAFFT+back-translate) is the worst on broken ORFs — it
   emits all-gap rows silently; worst possible failure mode.
3. MACSE fails to finish at n=512 within 90 min (0/10 reps); genomsa
   does it in ~80 s.

## Parameter sweep (fs cells, `sim_precision_{r2b120,r4b240,fs15,fs60}`)

| config | fs-3 | fs-1 | fs-0.5 |
|---|---|---|---|
| base (refine1, band40, fs30) | 0.174 | 0.058 | 0.021 |
| refine2, band120 | 0.184 | 0.066 | 0.022 |
| refine4, band240 | 0.193 | 0.071 | 0.018 |
| refine2, fs15 | **0.153** | 0.060 | 0.020 |
| refine2, fs60 | 0.445 | 0.094 | 0.030 |

**Parameter space is exhausted**: more refine passes / wider bands give
nothing; only cheaper fs cost helps marginally. The residual gap vs
MACSE is structural — stage-1 progressive alignment is frame-blind and
creates damage the banded stage-2 refine cannot reach. Phase-2 options:
fs-aware moves inside the progressive DP, or an outer
refine->re-tokenize->re-align loop.

## Speed (this benchmark, CPU, n=128 L=1500)

genomsa/lf ~15 s median; threestep 86 s; MACSE 909 s median, max 89 min.
On the real 31-locus panel: genomsa ~41 min total, lf ~14 min, MACSE
~49 h (legacy timed-out records).

## Caveats

- Simulator is ours; lf shares its codon-frame assumptions (home-court
  risk mitigated by adversarial cells where lf neither wins nor loses).
- FastTree proxy, not the production partitioned-ML estimator.
- MACSE ran with -max_refine_iter 0 (its own refinement off) — its true
  ceiling is higher than measured here.

## Full-scale RF vs Upham (measured, 2026-09-18)

RF distances vs `upham_pruned_4353.tre`, 3786 shared taxa
(`scripts/rf_distance.py`):

| Tree | RF | RF_norm |
|---|---|---|
| backbone, supermatrix base | 2112 | 0.279 |
| backbone, supermatrix codon/MACSE | 2268 | 0.300 |
| backbone, supermatrix genomsa | 2050 | 0.271 |
| lf chain s3 (mid-run) | 2040 | 0.270 |
| lf chain s1 (mid-run) | 2086 | 0.276 |
| genomsa vs codon (pairwise) | 2690 | 0.309 |

All our backbones sit at RF_norm ~0.27-0.30 vs Upham — comparable
divergence to independent analyses of these data. Our own methods
disagree with each other (0.309) more than either disagrees with Upham.
lf chains still running; numbers are snapshots of intermediate trees.

Cross-matrix logL (-te, au_test): Upham is the *worst* candidate
topology on the codon matrix (see conversation notes); lf row pending
(te_lf_1,2 done ~3.5h, te_lf_0 resubmitted on kbs after sixhour cap).

## Pilot curation (prefilter, measured)

8-gene x 200-seq pilot, ACC_CSV + MEDIAN_MULT=20:
APOB dropped 1 (189 kb BAC clone AC139752.4), COI dropped 68
(complete mitogenomes ~16 kb), other genes clean. Root cause of the
MACSE silent rc=1: raw genomic records killing the pairwise-distance
phase. MACSE completed 7/8 loci on the filtered set (CYTB needs
>90 min timeout); -seq_lr routes surviving >10 kb records.

## Cross-matrix topology logL (-te, complete 3x3, 2026-09-18)

Optimized logL per candidate topology on each supermatrix
(3786 shared taxa, partitioned; one iqtree2 -te per cell):

| Matrix \ Topology | genomsa | MACSE | Upham |
|---|---|---|---|
| base  | **-3,704,902** | -3,718,137 | -3,711,967 |
| codon | -3,977,857 | **-3,954,166** | -3,983,792 |
| lf    | **-3,888,076** | -3,892,225 | -3,900,234 |

The diagonal wins in all three matrices. On the lf matrix the
genomsa-derived topology beats MACSE by dL 4,149 and Upham by
dL 12,158. Upham's topology never wins any matrix; on codon and lf it
is the worst candidate.
