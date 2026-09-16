#!/usr/bin/env python3
"""Prepare a VBPI pilot: taxon-subset FASTA + induced support trees.

Takes a supermatrix FASTA, samples/keeps N taxa (stratified shuffle with
seed), writes subset FASTA, and prunes each candidate tree (newick files
or a multi-line .trees/.ufboot file) to the subset taxa.

Usage:
  prep_pilot.py --fasta sm.fasta --n 256 --seed 0 \
      --trees cand1.tre cand2.ufboot ... --outdir pilot_n256
"""
import argparse
import random

from Bio import SeqIO
from ete3 import Tree

p = argparse.ArgumentParser()
p.add_argument('--fasta', required=True)
p.add_argument('--n', type=int, required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--trees', nargs='+', required=True)
p.add_argument('--outdir', required=True)
a = p.parse_args()

recs = {r.id: r for r in SeqIO.parse(a.fasta, 'fasta')}
ids = sorted(recs)
random.Random(a.seed).shuffle(ids)
keep = ids[:a.n]
print(f'{len(ids)} taxa -> subset of {len(keep)}')

import os
os.makedirs(a.outdir, exist_ok=True)
with open(f'{a.outdir}/sub.fasta', 'w') as f:
    for i in keep:
        f.write(f'>{i}\n{recs[i].seq}\n')

keep_set = set(keep)
n_out = 0
with open(f'{a.outdir}/support.trees', 'w') as out:
    for tf in a.trees:
        for line in open(tf):
            line = line.strip()
            if not line or not line.startswith('('):
                continue
            t = Tree(line)
            leaves = set(t.get_leaf_names())
            if not leaves & keep_set:
                continue
            t.prune([l for l in leaves & keep_set],
                    preserve_branch_length=False)
            out.write(t.write(format=9) + '\n')
            n_out += 1
print(f'wrote {n_out} induced support trees -> {a.outdir}/support.trees')
