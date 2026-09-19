#!/usr/bin/env python3
"""Prune the Upham MamPhy consensus tree to the taxa present in our
supermatrix (or any newick tree), and optionally prune our tree to the
same intersection so iqtree2 -rf can compare them.

Upham tips look like 'Zaglossus_bartoni_TACHYGLOSSIDAE_MONOTREMATA' or
'_Anolis_carolinensis'; we normalize to 'Genus_species'. Our tree tips are
'Genus_species' or unmapped accessions (reported, skipped).

Usage:
  upham_prune.py --upham upham_ndexp_mcc.tre --ours supermatrix.fasta \
      --out upham_pruned.tre [--ours-tree backbone.treefile --ours-out backbone_pruned.tre]
"""
import argparse, re, sys

def our_tips_from_fasta(path):
    tips = []
    for line in open(path):
        if line.startswith(">"):
            tips.append(line[1:].split()[0])
    return tips

def our_tips_from_newick(path):
    return [t.name for t in Phylo.read(path, "newick").get_terminals()]

def norm_upham(label):
    """'Zaglossus_bartoni_TACHYGLOSSIDAE_MONOTREMATA' -> 'Zaglossus_bartoni'
       '_Anolis_carolinensis' -> 'Anolis_carolinensis'"""
    s = label.strip().lstrip("_").strip("'\"")
    # strip _FAMILY_ORDER suffix: two trailing ALL-CAPS underscore fields
    toks = s.split("_")
    while len(toks) > 2 and toks[-1].isupper():
        toks.pop()
    return "_".join(toks)

def upham_tips_and_tree(nexus_path):
    """Parse NEXUS taxlabels + the translate/newick tree via Bio.Phylo."""
    return Phylo.read(nexus_path, "nexus")

def prune(tree, keep):
    for t in tree.get_terminals():
        if t.name not in keep:
            tree.prune(t)
    return tree

def load_names_dmp(path):
    """NCBI names.dmp -> (name -> {taxids}, taxid -> scientific_name)."""
    name2id, sci = {}, {}
    for line in open(path, errors="replace"):
        f = [x.strip() for x in line.split("|")]
        if len(f) < 4:
            continue
        tid = int(f[0])
        name2id.setdefault(f[1], set()).add(tid)
        if f[3] == "scientific name":
            sci[tid] = f[1]
    return name2id, sci

def to_species_key(name):
    """'Genus species ...' -> 'Genus_species' (first two tokens)."""
    t = name.split()
    return f"{t[0]}_{t[1]}" if len(t) >= 2 else None

if __name__ == "__main__":
    from Bio import Phylo
    ap = argparse.ArgumentParser()
    ap.add_argument("--upham", required=True)
    ap.add_argument("--ours", required=True, help="supermatrix fasta or newick")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ours-tree", help="our backbone newick to prune too")
    ap.add_argument("--ours-out")
    ap.add_argument("--names-dmp", help="NCBI names.dmp for synonym resolution")
    a = ap.parse_args()

    if a.ours.endswith((".tre", ".nwk", ".treefile")):
        our = our_tips_from_newick(a.ours)
    else:
        our = our_tips_from_fasta(a.ours)
    our_set = set(our)
    print(f"our taxa: {len(our)} ({len(our_set)} unique)")

    tree = upham_tips_and_tree(a.upham)
    raw = [t.name for t in tree.get_terminals()]
    # rename Upham tips in place to normalized binomials
    seen = {}
    for t in tree.get_terminals():
        t.name = norm_upham(t.name)
        seen[t.name] = seen.get(t.name, 0) + 1
    dups = [k for k, v in seen.items() if v > 1]
    print(f"upham tips: {len(raw)} normalized ({len(dups)} dup labels)")

    keep = our_set & set(seen)
    unmatched = our_set - set(seen)

    # taxid-level synonym resolution for names that don't string-match
    ours_relabel = {}   # our_tip -> upham binomial it maps to
    if a.names_dmp:
        name2id, sci = load_names_dmp(a.names_dmp)
        # taxid -> upham binomial (first species-level name variant wins)
        tid2upham = {}
        for up_name in seen:
            up_sp = up_name.replace("_", " ")
            for tid in name2id.get(up_sp, ()):
                tid2upham.setdefault(tid, up_name)
        resolved = 0
        for x in sorted(unmatched):
            sp = x.replace("_", " ")
            tids = name2id.get(sp, set())
            hits = {tid2upham[t] for t in tids if t in tid2upham}
            if len(hits) == 1:
                ours_relabel[x] = hits.pop()
                resolved += 1
        print(f"synonym-resolved: {resolved} of {len(unmatched)} unmatched")
        keep |= set(ours_relabel)

    acc = [x for x in unmatched - set(ours_relabel)
           if re.match(r"^[A-Z]{1,3}[0-9]", x)]
    print(f"intersection: {len(keep)}  | ours unmatched: "
          f"{len(unmatched - set(ours_relabel))} (accession-like: {len(acc)})")
    if len(keep) < 10:
        sys.exit("intersection too small -- check name normalization")

    prune(tree, keep)
    Phylo.write(tree, a.out, "newick")
    print(f"wrote {a.out}: {len(keep)} tips")

    if a.ours_tree and a.ours_out:
        ot = Phylo.read(a.ours_tree, "newick")
        n0 = len(ot.get_terminals())
        for t in ot.get_terminals():
            if t.name in ours_relabel:
                t.name = ours_relabel[t.name]  # rename to upham binomial
            elif t.name not in keep:
                ot.prune(t)
        Phylo.write(ot, a.ours_out, "newick")
        print(f"wrote {a.ours_out}: {n0} -> {len(keep)} tips")
