#!/usr/bin/env python3
"""Build mini-supermatrices for fast Upham-replication experiments.

Subsets existing per-locus alignments (rows -> sampled SPECIES, all
columns) from each alignment variant and concatenates into a supermatrix
+ IQ-TREE partition file, per replicate.

Row ids in the per-locus alignments are GenBank accessions; species names
are recovered from genes_raw headers ("ACC.V Genus species ..."), keeping
the accession with fewest ambiguous sites per species (same rule as
pai_supermatrix.py).

Usage:
  mini_backbone.py --outdir DIR --reps 5 --ntaxa 300 --nloci 8 \
      --upham upham.tre --raw DIR_genes_raw \
      --aln macse:DIR base:DIR lf:DIR
"""
import argparse
import os
import random
import re

p = argparse.ArgumentParser()
p.add_argument('--outdir', required=True)
p.add_argument('--reps', type=int, default=5)
p.add_argument('--ntaxa', type=int, default=300)
p.add_argument('--nloci', type=int, default=8)
p.add_argument('--upham', required=True)
p.add_argument('--raw', required=True, help='genes_raw dir')
p.add_argument('--aln', nargs='+', required=True,
               help='name:dir with per-locus fasta alignments')
p.add_argument('--mincov', type=int, default=6,
               help='min loci a species must have (missing -> gap rows)')
a = p.parse_args()

SKIP_TOKENS = {'sp.', 'sp', 'cf.', 'cf', 'aff.', 'aff', 'nr', 'x'}


def read_fasta(path):
    d, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith('>'):
            name = line[1:].split()[0]
            d[name] = []
        elif line and name:
            d[name].append(line)
    return {k: ''.join(v) for k, v in d.items()}


def read_fasta_full_header(path):
    d, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith('>'):
            name = line[1:]
            d[name] = []
        elif line and name:
            d[name].append(line)
    return {k: ''.join(v) for k, v in d.items()}


def species_of(header):
    toks = header.split()
    if len(toks) < 3:
        return None
    g, s = toks[1], toks[2]
    if s.lower() in SKIP_TOKENS or not g[0].isupper():
        return None
    return (re.sub(r'[^A-Za-z_]', '', g) + '_' +
            re.sub(r'[^A-Za-z]', '', s))


def leaves_from_newick(path):
    s = open(path).read()
    s = re.sub(r'\[[^\]]*\]', '', s)          # strip [&...] annotations
    out = []
    for tok in re.split(r'[(),:;]', s):
        tok = tok.strip().strip("'\"")
        if tok and not re.match(r'^[\d.Ee+-]+$', tok):
            out.append(tok)
    return set(out)


# ---- accession -> species map from raw headers ----
variants = {}
for spec in a.aln:
    name, d = spec.split(':', 1)
    variants[name] = {f.split('.')[0]: os.path.join(d, f)
                      for f in os.listdir(d) if f.endswith(('.fasta', '.fa'))}
loci_all = sorted(set.intersection(*[set(v) for v in variants.values()]))

acc2sp = {}
for g in loci_all:
    rp = os.path.join(a.raw, f'{g}.fasta')
    if not os.path.exists(rp):
        continue
    for h in read_fasta_full_header(rp):
        acc = h.split()[0]
        acc2sp.setdefault(acc, species_of(h))

rng0 = random.Random(0)
loci = sorted(rng0.sample(loci_all, min(a.nloci, len(loci_all))))
print(f'loci ({len(loci)}): {loci}')

# ---- load alignments, species-keyed, best accession per species ----
def badness(seq):
    return sum(1 for c in seq if c not in 'ACGT')


alns = {}   # variant -> locus -> {species: seq}
for vname, files in variants.items():
    alns[vname] = {}
    for g in loci:
        best = {}
        for acc, seq in read_fasta(files[g]).items():
            sp = acc2sp.get(acc)
            if not sp:
                continue
            if sp not in best or badness(seq) < badness(best[sp]):
                best[sp] = seq
        alns[vname][g] = best

upham_leaves = leaves_from_newick(a.upham)
# species must be in Upham and have >= mincov loci in EVERY variant
cov = {}
for sp in upham_leaves:
    ok = True
    for vname in variants:
        c = sum(sp in alns[vname][g] for g in loci)
        if c < a.mincov:
            ok = False
            break
    if ok:
        cov[sp] = True
shared = sorted(cov)
print(f'shared species (>= {a.mincov}/{len(loci)} loci, all variants, '
      f'Upham): {len(shared)}')

ntaxa = min(a.ntaxa, len(shared))
for rep in range(a.reps):
    r = random.Random(1000 + rep)
    taxa = sorted(r.sample(shared, ntaxa))
    rd = os.path.join(a.outdir, f'rep{rep}')
    for vname in variants:
        vd = os.path.join(rd, vname)
        os.makedirs(vd, exist_ok=True)
        parts, pos = [], 1
        with open(os.path.join(vd, 'supermatrix.fasta'), 'w') as sf:
            for t in taxa:
                sf.write(f'>{t}\n')
                sf.write(''.join(alns[vname][g].get(t) or
                                 '-' * len(next(iter(alns[vname][g].values())))
                                 for g in loci) + '\n')
            for g in loci:
                w = len(next(iter(alns[vname][g].values())))
                parts.append(f'DNA, {g} = {pos}-{pos + w - 1}')
                pos += w
        with open(os.path.join(vd, 'partitions.txt'), 'w') as pf:
            pf.write('\n'.join(parts) + '\n')
        with open(os.path.join(vd, 'taxa.txt'), 'w') as tf:
            tf.write('\n'.join(taxa) + '\n')
    print(f'rep{rep}: {ntaxa} taxa x {pos - 1} sites -> {rd}')
print('done')
