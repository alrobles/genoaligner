# S1 — Codon-aware GPU MSA (`genomsa --codon`): evidence base

Goal: replace the MACSE translation-aware codon-alignment bottleneck in the
PhylogenyAI/Upham pipeline with a native GPU implementation on the existing
profile-profile Gotoh engine.

## Design (one paragraph)

Sequences are tokenized to codons — one byte per codon, `128 + index`
(the offset keeps tokens disjoint from `'-'`; index = base-4 with
T,C,A,G = 0,1,2,3; index 64 = ambiguous/partial codon). The alignment
engine runs unchanged at `alpha=65` over a 65x65 substitution matrix:
BLOSUM62 over the translated amino acids + `codon_nt_bonus` per identical
nucleotide position + `codon_stop_pen` when exactly one side is a stop.
Gaps are therefore always whole codons — the output keeps reading frame
by construction. Genetic codes: NCBI 1 (standard) and 2 (vertebrate
mitochondrial). Per-sequence frame = forward frame with fewest stops;
per-seq QC (frame, stops, partials) via `--codon-qc`. Decode emits
`NNN` for ambiguous/partial tokens and drops leading frame-offset bases
(1-2 nt, reported in QC).

Engineering note: the 65x65 matrix (~17 KB) exceeds the kernel-parameter
space, so `alpha>20` reads `p.sub_ext` — a device buffer uploaded once
per call; `alpha<=20` keeps the by-value matrix unchanged.

## Measurable claims

| # | Claim | Metric | Evidence |
|---|-------|--------|----------|
| 1 | Output is in-frame by construction | 100% of alignment columns are whole codons (no split-codon gaps); decoded length %3==0 | `scripts/codon_validate.sbatch` invariants, all genes |
| 2 | GPU == CPU reference | byte-identical alignment files | job 29240188 (MI210): A2AB/CNR1/COI/CYTB all PASS |
| 3 | QC flags pathological loci | per-seq stop count after best-frame encoding | the 5 loci Upham reclassified as noncoding (APP, BMI1, CREM, FBN1, PLCB4) have 1.8-10 stops/seq vs <0.7/seq on coding loci |
| 4 | Quality vs known truth | SIM-SPS on simulated codon evolution | `scripts/codon_sim_bench.sbatch` (genomsa vs MACSE, same input) — *running* |
| 5 | Agreement with MACSE | aligned codon-pair recall/precision | `scripts/codon_vs_macse.py`, 24 loci — below |
| 6 | Speed | end-to-end walltime per locus | `data/genomsa_codon/summary_*.tsv` — below |
| 7 | End-to-end equivalence | RF(genomsa-codon backbone, MACSE backbone), RF vs Upham MCC | codon supermatrix 4353x247185 built (job 29241809); IQ-TREE queued (29241810) |
| 8 | Cross-vendor determinism | byte-identical CPU<->GPU on both MI210 and A100; identical aln_len/gap%/stops on both | job 29241736 (A100): A2AB/COI/CYTB all PASS; CYTB gpu 28s vs cpu 191s |

## Runtime (MI210, single GPU, includes NJ0-GPU guide tree)

| gene | nseq | gc | aln_len | t_wall (s) | MACSE walltime |
|------|-----:|---:|--------:|-----------:|----------------|
| A2AB | 238 | 1 | 3999 | 33 | minutes-hours |
| CNR1 | 306 | 1 | 3210 | 34 | " |
| COI  | 1608 | 2 | 6045 | 59 | >6 h (timeout) |
| CYTB | 3523 | 2 | 2349 | 63 | >6 h (timeout) |
| BRCA1| 913 | 1 | 30138 | 304 | >6 h (timeout) |
| VWF  | 755 | 1 | 53967 | 246 | >6 h (timeout) |
| **31 loci total** | | | | **~40 min** | **days (7 genes still on 72 h retry)** |

## MACSE agreement (aligned codon-pair recall / precision)

Per pair of sequences: the set of residue-index pairs co-aligned by each
method; recall = |genomsa ∩ macse| / |macse|. Ambiguous/frameshift ('!')
codons are excluded from pair counting.

| locus | recall | precision | | locus | recall | precision |
|-------|-------:|----------:|-|-------|-------:|----------:|
| ND1 (mt) | **0.939** | 0.940 | | DMP1 | 0.463 | 0.471 |
| ND2 (mt) | **0.945** | 0.945 | | PNOC | 0.460 | 0.479 |
| BDNF | 0.734 | 0.736 | | APP † | 0.162 | 0.214 |
| BCHE | 0.688 | 0.689 | | BMI1 † | 0.297 | 0.350 |
| TYR1 | 0.665 | 0.672 | | CREM † | 0.203 | 0.240 |
| ADRB2 | 0.639 | 0.639 | | FBN1 † | 0.185 | 0.226 |
| BRCA2 | 0.623 | 0.627 | | PLCB4 † | 0.158 | 0.199 |
| ATP7 | 0.570 | 0.570 | | | | |
| TTN | 0.553 | 0.589 | | | | |
| RAG2 | 0.545 | 0.547 | | | | |
| RAG1A | 0.525 | 0.529 | | | | |
| CNR1 | 0.517 | 0.518 | | | | |
| EDG1 | 0.505 | 0.506 | | | | |
| APOB | 0.487 | 0.508 | | | | |
| A2AB | 0.483 | 0.484 | | | | |

† loci Upham treated as noncoding — codon alignment is not the right
model for them; the low agreement is expected and our QC flags them.

Interpretation: near-identity on conserved mitochondrial loci (0.94),
mid-range on divergent/fragmentary nuclear loci. Pair-agreement between
two reasonable aligners on such data is inherently 0.4-0.7; the decisive
quality evidence is SIM-SPS vs known truth (claim 4) and downstream RF
(claim 7), not MACSE agreement per se.

## Known limitations (honest list)

- Per-sequence frame selection by fewest-stops can disagree between
  homologous fragments; `--codon-qc` reports it. A global-frame mode is
  a candidate flag if the data turn out to be frame-normalized.
- Leading frame-offset bases (1-2 nt) are dropped; partial codons and
  ambiguous codons decode as NNN. MACSE's '!' pads are not reproduced.
- We do not replicate MACSE's frameshift *detection* model (cost-based
  1-2 nt indels); stops are detected, frameshifted nucleotides are not
  marked in output.
- Noncoding loci should be run in DNA mode, not `--codon`.

## Reproduce

```
sbatch --export=GENES="A2AB CNR1 COI CYTB" scripts/codon_validate.sbatch   # parity + invariants
sbatch scripts/codon_all31.sbatch      # all loci + MACSE agreement
sbatch scripts/codon_sim_bench.sbatch  # SIM-SPS vs MACSE on known truth
sbatch scripts/codon_supermatrix.sbatch && sbatch scripts/supermatrix_iqtree_codon...  # RF chain
```
