#!/usr/bin/env python3
"""Build mini-supermatrices for fast Upham-replication experiments.

Subsets existing per-locus alignments (rows -> sampled taxa, all columns)
from each alignment variant and concatenates into a supermatrix +
IQ-TREE partition file, per replicate.

Usage:
  mini_backbone.py --outdir DIR --reps 5 --ntaxa 300 --nloci 8 \
      --upham upham.tre --aln macse:DIR base:DIR lf:DIR
"""
import argparse
import os
import random

p = argparse.ArgumentParser()
p.add_argument('--outdir', required=True)
p.add_argument('--reps', type=int, default=5)
p.add_argument('--ntaxa', type=int, default=300)
p.add_argument('--nloci', type=int, default=8)
p.add_argument('--upham', required=True)
p.add_argument('--aln', nargs='+', required=True,
               help='name:dir with per-locus fasta alignments')
a = p.parse_args()


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


def leaves_from_newick(path):
    s = open(path).read()
    out, cur = [], ''
    for c in s:
        if c in '(),:;':
            if cur:
                out.append(cur)
            cur = ''
        elif c not in ' \t\n':
            cur += c
        else:
            cur = ''
    return set(x.strip("'").strip('"') for x in out if not
               x.replace('.', '').replace('-', '').isdigit())


# discover per-variant locus files
variants = {}
for spec in a.aln:
    name, d = spec.split(':', 1)
    variants[name] = {f.split('.')[0]: os.path.join(d, f)
                      for f in os.listdir(d) if f.endswith(('.fasta', '.fa'))}
loci = sorted(set.intersection(*[set(v) for v in variants.values()]))
rng = random.Random(0)
loci = sorted(rng.sample(loci, min(a.nloci, len(loci))))
print(f'loci ({len(loci)}): {loci}')

alns = {vname: {g: read_fasta(variants[vname][g]) for g in loci}
        for vname in variants}

upham_leaves = leaves_from_newick(a.upham)
shared = set(upham_leaves)
for g in loci:
    for vname in variants:
        shared &= set(alns[vname][g])
shared = sorted(shared)
print(f'shared taxa across {len(loci)} loci x {len(variants)} variants '
      f'+ Upham: {len(shared)}')

ntaxa = min(a.ntaxa, len(shared))
for rep in range(a.reps):
    r = random.Random(1000 + rep)
    taxa = sorted(r.sample(shared, ntaxa))
    rd = os.path.join(a.outdir, f'rep{rep}')
    os.makedirs(rd, exist_ok=True)
    for vname in variants:
        vd = os.path.join(rd, vname)
        os.makedirs(vd, exist_ok=True)
        parts, pos = [], 1
        with open(os.path.join(vd, 'supermatrix.fasta'), 'w') as sf:
            for t in taxa:
                sf.write(f'>{t}\n')
                sf.write(''.join(alns[vname][g][t] for g in loci) + '\n')
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
