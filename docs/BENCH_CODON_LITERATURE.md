# Credibility plan for `genomsa --codon`: what the literature demands

Literature sweep (Sep 2026). Goal: design a benchmark that survives
reviewer scrutiny — not "agrees with MACSE" but measurable correctness.

## 1. How codon-aware aligners are actually evaluated in the literature

**MACSE's own papers are weak on benchmarks.** Ranwez et al. 2011
(PLOS ONE e22594) validates on three *biological case studies* — the
cetartiodactyl AMBN pseudogene alignment, kiwi/chicken olfactory
receptors (93 functional + 18 pseudogenes), and NGS-read placement —
all visual/case-based, no controlled truth. Ranwez et al. 2018 (MBE
msy159, v2) is a toolkit paper; Delsuc & Ranwez 2020 (MACSE_BARCODE)
validates by *rediscovering manually-curated BOLD errors*.
**No head-to-head truth-based benchmark of MACSE exists** — a rigorous
simulation evaluation is a genuine gap we can fill (and cite).

**The rigorous tradition lives in the PRANK/Goldman-lab lineage:**

- Löytynoja & Goldman 2008 (Science 320:1632): phylogeny-aware
  simulation with tracked indel events; PRANK wins by distinguishing
  insertion from deletion history. `-F` (trust guide tree) best under
  high indel rates.
- Fletcher & Yang 2009/2010 (INDELible, MBE msp098; msq115):
  simulate codon evolution along a *known tree* under MG94-style codon
  models with varying dN/dS + indels with realistic length
  distributions; metric = alignment accuracy vs true alignment +
  **downstream branch-site positive-selection test FPR/power** — the
  money metric: misaligned codon columns create fake selection.
- Jordan & Goldman 2012 (MBE msr272): alignment-error → sitewise ω
  false positives; PRANK-codon least affected; GUIDANCE filtering
  best. Established that **codon-column misalignment is THE driver of
  false-positive selection inference**.
- Markova-Raina & Petrov 2011 (Genome Res 21:863): 12-Drosophila
  genomes, 48-82% of inferred positive-selection sites are
  misalignment artifacts; PRANK best but still 50% FPR.
- Markova-Raina & Petrov-style empirical evaluation + Dessimoz-lab
  filtering work (Tan et al. 2015): alignment→tree accuracy measured
  by RF/topological tests on simulated + empirical data.
- ASR angle (msy055): alignment quality correlates with ancestral
  reconstruction accuracy — relevant to the arfun-paper lineage.

**Frameshift detection** has its own evaluation literature
(GeneTack/MetaGeneTack/AlignWise): sensitivity+specificity on
known-position simulated 1-2 nt indels in real CDS — (Sn+Sp)/2.

## 2. What reviewers will ask (and whether we have it)

| question | the literature's instrument | our status |
|---|---|---|
| Correct vs known truth? | simulated codon evolution w/ tracked column ids, SP/TC | **partially** — custom sim works (SPS grid done) but reviewers expect codon-model subs (MG94 dN/dS), not raw-nt subs |
| vs established tools? | MACSE, PRANK-codon, 3-step back-translation | MACSE done (init+refined); **PRANK missing** — in bioconda, must add |
| Handles real frameshifts/pseudogenes? | inject 1-2nt indels + stop codons into real CDS | **missing** — our sim does whole-codon indels only |
| Preserves codon structure? | in-frame output, no split codons | done (by construction) |
| QC flags match biology? | concordance with known pathological loci | done (5 Upham noncoding loci) |
| Downstream utility? | tree RF vs reference, dN/dS FPR | RF chain armed (iqtree running); dN/dS FPR not done |
| Scale? | n=10^4-10^5 walltime | 31 loci/40min done; CYTB 3523 seqs/63s; MACSE_BARCODE enrich = 140h CPU for Mammalia COI — our scale claim slot |

## 3. Concrete benchmark build-out (ranked by credibility-per-effort)

**B1 — Codon-model simulation upgrade (highest value, ~1 day).**
Extend `msa_sim_truth.py` to evolve at the *codon* level: MG94/GY
substitutions with per-site dN/dS drawn from a distribution (purifying
ω<1 mostly, neutral ω=1, bursts ω>1) + whole-codon indels (done) +
**optional 1-2nt frameshift indels on a pseudogene fraction** of
sequences. This directly mirrors Fletcher&Yang and gives: SPS + TC +
frameshift Sn/Sp + pseudogene robustness in one instrument. The
current nt-sub model is defensible but codon-model subs are what this
literature recognizes.

**B2 — PRANK `-codon -F` comparator (cheap, in bioconda).**
Add prank to the sim grid: `prank -d=sim_in.fasta -codon -F
-t=sim_true_tree.nwk` (feeding it the TRUE guide tree is the
literature-fair comparison and isolates alignment quality from tree
error). PRANK is CPU/slow too — run on the smaller cells.

**B3 — Three-step baseline (TranslatorX-style).**
mafft --auto on AA translations → back-translate to codon alignment →
SPS. This is the naive alternative; expected to break on our sim's
in-frame stops (which is exactly MACSE's sales pitch — we should show
we handle what 3-step can't).

**B4 — Curated reference: MACSE_BARCODE alignments on Zenodo.**
The BOLD-derived COI Mammalia reference (117k seqs) is a published
curated alignment — align a subsample with genomsa --codon, score
against their reference columns. External truth, real data.

**B5 — dN/dS FPR (the Goldman-lineage money metric).**
Simulate under strict purifying selection (ω<1 everywhere, no positive
selection), align with each tool, run a sitewise test (even a simple
counting test or PAML M8a vs M8 if available). Claim: our alignments
produce no more false-positive selection calls than the true
alignment. *Defer unless phylogenyAI needs selection analyses.*

**B6 — Frameshift injection on real CDS (cheap).**
Take QC-pass real genes, inject k known 1-nt deletions in k random
sequences, check our QC / output handles it and MACSE `!` marks the
same spots → Sn/Sp table. Complements the stop-concordance we have.

## 4. What we already have that the MACSE papers never did

- Controlled-truth SPS vs MACSE-init AND MACSE-refined on the same
  input (genomsa wins 4/6 cells, ties at saturation; refined MACSE
  3.6h/cell vs our ~35-311s).
- Byte-identical CPU↔GPU on two vendors.
- 31 real loci end-to-end + curated-QC concordance.
- Pipeline-level: codon supermatrix 4353×247185 → IQ-TREE → RF vs
  MACSE and Upham-MCC arms (in flight).

## 5. Honest framing for the milestone/paper

Claim: "GPU codon-aware MSA matching or exceeding MACSE-class accuracy
on simulated truth at 20-300x lower walltime, validated end-to-end on
a 31-locus mammalian phylogenomics pipeline." Do NOT claim: parity
with refined MACSE everywhere (saturation regime is genuinely hard for
everyone), frameshift-marking equivalence (we detect stops, not `!`
markers), or iterative-refinement quality (we are single-pass
progressive — MACSE's refine is a different, expensive instrument).

## References

- Ranwez et al. 2011 PLOS ONE e22594 (MACSE); Ranwez et al. 2018 MBE
  msy159 (v2); Delsuc & Ranwez 2020 (MACSE_BARCODE)
- Fletcher & Yang 2009 MBE msp098 (INDELible); 2010 MBE msq115
- Löytynoja & Goldman 2008 Science 320:1632 (PRANK)
- Jordan & Goldman 2012 MBE msr272 (alignment error -> selection FPR)
- Markova-Raina & Petrov 2011 Genome Res 21:863 (Drosophila FPR)
- Tan et al. 2015 (filtering -> tree accuracy, RF methodology)
- Kosiol, Holmes & Goldman 2007 MBE 24:1464 (empirical codon model —
  PRANK's codon matrix)
