#!/usr/bin/env python3
"""Simulation benchmark for genomsa: evolve sequences with a KNOWN true
alignment, then measure how much of it the MSA recovers.

Model: random ancestor (~1500 bp) evolved along a balanced binary tree;
per-branch substitutions and indels. Column identity is tracked through a
GLOBAL ordered column list: the ancestor owns columns 0..L-1; every
insertion event creates fresh column ids placed in the global order
immediately after its anchor column (the residue it lands behind in that
leaf), which keeps the global order consistent with every leaf's internal
sequence order. The TRUE MSA renders each sequence over that global order.

Metric: column-SPS -- for every true column, the residue pairs it aligns;
fraction reproduced in the candidate MSA (the standard sum-of-pairs
criterion of simulation benchmarks).

Usage: msa_sim_truth.py <msa.fasta|-> [n_leaves] [seed] [sub_rate] [indel_rate]
Writes sim_in.fasta / sim_true.fasta; prints 'SIM-SPS x.xxxx' when given an
alignment to score.
"""
import sys, random

def simulate(n_leaves=64, L=1500, seed=1, sub_rate=0.06, indel_rate=0.02):
    rng = random.Random(seed)
    BASES = "ACGT"
    order = list(range(L))             # global true column order (col ids)
    next_id = [L]
    anc = [(rng.choice(BASES), i) for i in range(L)]

    def evolve(seq, n_indels, n_subs):
        s = list(seq)
        for _ in range(n_subs):
            i = rng.randrange(len(s))
            b = rng.choice([x for x in BASES if x != s[i][0]])
            s[i] = (b, s[i][1])
        for _ in range(n_indels):
            ln = 1 + int(rng.expovariate(1 / 2.5))
            pos = rng.randrange(len(s) + 1)
            if rng.random() < 0.5:
                new = list(range(next_id[0], next_id[0] + ln))
                next_id[0] += ln
                blk = [(rng.choice(BASES), c) for c in new]
                if pos > 0:
                    oi = order.index(s[pos - 1][1]) + 1
                else:
                    oi = 0
                order[oi:oi] = new
                s[pos:pos] = blk
            else:
                del s[pos:pos + ln]
        return s

    def branch(seq):
        n_sub = max(1, round(len(seq) * sub_rate * (0.75 + 0.5 * rng.random())))
        n_ind = round(len(seq) * indel_rate * (0.5 + rng.random()))
        return evolve(seq, n_ind, n_sub)

    def split(seq, k):
        if k == 1:
            return [seq]
        mid = k // 2
        return split(branch(seq), mid) + split(branch(seq), k - mid)

    leaves = split(anc, n_leaves)
    # true guide tree (balanced splits mirror the simulation topology):
    # leaves are written in split order, so the tree is
    #   split(k) = (split(k//2), split(k-k//2)) over consecutive leaf ids
    def true_nwk(lo, k):
        if k == 1:
            return f"s{lo}"
        mid = k // 2
        return f"({true_nwk(lo, mid)},{true_nwk(lo + mid, k - mid)})"

    tree_nwk = true_nwk(0, n_leaves) + ";\n"
    colidx = {c: i for i, c in enumerate(order)}
    true_rows = []
    for s in leaves:
        row = ["-"] * len(order)
        for b, c in s:
            row[colidx[c]] = b
        true_rows.append("".join(row))
    leaf_seqs = ["".join(b for b, _ in s) for s in leaves]
    return leaf_seqs, true_rows, tree_nwk

def sps(true_rows, got_rows):
    n = len(true_rows)
    same = tot = 0
    def resmap(row):                    # residue ordinal -> column
        m = {}; r = 0
        for j, c in enumerate(row):
            if c != "-":
                m[r] = j; r += 1
        return m
    tm = [resmap(r) for r in true_rows]
    gm = [resmap(r) for r in got_rows]
    for a in range(n):
        for b in range(a + 1, n):
            inv_b = {v: k for k, v in tm[b].items()}
            for ra, ca in tm[a].items():
                rb = inv_b.get(ca)
                if rb is None:
                    continue
                tot += 1
                if gm[a].get(ra) == gm[b].get(rb):
                    same += 1
    return same / tot if tot else float("nan")

if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    sub = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
    ind = float(sys.argv[5]) if len(sys.argv) > 5 else 0.004
    seqs, true_rows, tree_nwk = simulate(n_leaves=n, seed=seed,
                                       sub_rate=sub, indel_rate=ind)
    with open("sim_in.fasta", "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">s{i}\n{s}\n")
    with open("sim_true.fasta", "w") as f:
        for i, s in enumerate(true_rows):
            f.write(f">s{i}\n{s}\n")
    with open("sim_true_tree.nwk", "w") as f:
        f.write(tree_nwk)
    if sys.argv[1] != "-":
        got = {}
        cur = None
        for line in open(sys.argv[1]):
            line = line.strip()
            if line.startswith(">"):
                cur = line[1:].split()[0]
                got[cur] = ""
            elif cur:
                got[cur] += line
        got_rows = [got[f"s{i}"] for i in range(n)]
        # sanity: ungapped got rows must equal the leaf sequences
        for i, s in enumerate(seqs):
            assert got_rows[i].replace("-", "") == s, f"row {i} corrupted"
        print(f"SIM-SPS {sps(true_rows, got_rows):.4f}  "
              f"width_true={len(true_rows[0])} width_got={len(got_rows[0])}")
    else:
        print(f"wrote sim_in.fasta / sim_true.fasta (n={n})")
