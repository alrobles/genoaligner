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
import re

from Bio import SeqIO
from ete3 import Tree

p = argparse.ArgumentParser()
p.add_argument('--fasta', required=True)
p.add_argument('--n', type=int, required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--trees', nargs='+', required=True)
p.add_argument('--nni', type=int, default=60,
               help='random NNI perturbations per induced tree to densify '
                    'the subsplit support set (VBPI sampling needs dense support)')
p.add_argument('--outdir', required=True)
a = p.parse_args()

rng = random.Random(a.seed + 7919)


def nni_perturb(tree):
    """One random NNI move: swap a child of internal node u with a child of
    internal child v. Returns a new tree, or None if no valid edge found."""
    t = tree.copy()
    internals = [n for n in t.traverse() if not n.is_leaf() and len(n.children) >= 2]
    rng.shuffle(internals)
    for u in internals:
        int_children = [c for c in u.children if not c.is_leaf() and len(c.children) >= 2]
        if not int_children or len(u.children) < 2:
            continue
        v = rng.choice(int_children)
        others = [c for c in u.children if c is not v]
        if not others:
            continue
        x, y = rng.choice(others), rng.choice(v.children)
        x.detach(); y.detach()
        u.add_child(y); v.add_child(x)
        return t
    return None

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
            for _ in range(3):  # strip nested NHX/BEAST [\[...\]] annotations
                line = re.sub(r"\[[^\[\]]*\]", "", line)
            t = Tree(line)
            leaves = set(t.get_leaf_names())
            if not leaves & keep_set:
                continue
            t.prune([l for l in leaves & keep_set],
                    preserve_branch_length=False)
            out.write(t.write(format=9) + '\n')
            n_out += 1
            for _ in range(a.nni):
                pt = nni_perturb(t)
                if pt is not None:
                    out.write(pt.write(format=9) + '\n')
                    n_out += 1
print(f'wrote {n_out} induced support trees -> {a.outdir}/support.trees')
