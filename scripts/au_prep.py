#!/usr/bin/env python3
"""Prepare a multi-tree file for IQ-TREE -z topology tests (AU/KH/SH).

Reads taxa from an alignment FASTA and a list of candidate trees, prunes
every tree to the taxa intersection (trees must share taxa with the
alignment for -z to accept them), and writes one Newick per line.

Usage: au_prep.py --fasta SM --out candidates.tre TREE [TREE...]
"""
import argparse
import re
import sys

from ete3 import Tree


def read_tree(path):
    # strip NHX/BEAST square-bracket annotations ete3 cannot parse
    nw = re.sub(r"\[[^\[\]]*\]", "", open(path).read())
    return Tree(nw, format=1)


def fasta_taxa(path):
    taxa = set()
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                taxa.add(line[1:].split()[0])
    return taxa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("trees", nargs="+")
    a = ap.parse_args()

    aln = fasta_taxa(a.fasta)
    kept = []
    common = set(aln)
    for p in a.trees:
        t = read_tree(p)
        names = {l.name for l in t.iter_leaves()}
        missing = names - aln
        if missing:
            print(f"WARN {p}: {len(missing)} tree taxa absent from alignment "
                  f"(e.g. {sorted(missing)[:3]})", file=sys.stderr)
        common &= names
        kept.append((p, t))

    print(f"alignment taxa: {len(aln)}; common across {len(kept)} trees: {len(common)}")
    with open(a.out, "w") as fh:
        for p, t in kept:
            t.prune(sorted(common), preserve_branch_length=True)
            fh.write(t.write(format=1) + "\n")
            print(f"  {p} -> {len(common)} leaves")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
