#!/usr/bin/env python3
"""Prune a FASTA alignment to the union of taxa in a trees file.

IQ-TREE -te requires every alignment row to appear in the evaluated
tree; candidate trees pruned to a shared taxon set therefore need a
matching pruned alignment. Columns (and thus partition schemes) are
unaffected.

Usage: sm_prune_taxa.py --fasta SM.fasta --trees candidates.tre --out OUT.fasta
"""
import argparse
import re
import sys

from Bio import SeqIO

p = argparse.ArgumentParser()
p.add_argument('--fasta', required=True)
p.add_argument('--trees', required=True,
               help='file with one newick per line; taxa = union of leaves')
p.add_argument('--out', required=True)
a = p.parse_args()

LEAF = re.compile(r"([A-Za-z0-9_.|/-]+)\s*(?::[0-9.eE+-]+)?\s*[,)]")

taxa = set()
for line in open(a.trees):
    line = line.strip()
    if not line.startswith('('):
        continue
    line = re.sub(r"\[[^\[\]]*\]", "", line)      # strip [\[...\]] NHX
    for m in LEAF.finditer(line):
        name = m.group(1).strip("'\"")
        if name and not re.fullmatch(r"[0-9.eE+-]+", name):
            taxa.add(name)

kept = dropped = 0
with open(a.out, 'w') as out:
    for r in SeqIO.parse(a.fasta, 'fasta'):
        if r.id in taxa:
            out.write(f'>{r.id}\n{r.seq}\n')
            kept += 1
        else:
            dropped += 1
print(f'{len(taxa)} tree taxa; kept {kept} rows, dropped {dropped}',
      file=sys.stderr)
