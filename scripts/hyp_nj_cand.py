#!/usr/bin/env python3
"""Experiment H2: do Poincare-embedding neighbourhoods recover NJ cherries
better than raw k-mer-D neighbourhoods?

nj_trace (C++) measures Recall@k of the exact NJ cherry inside per-round
candidate sets built from the *initial* D ("static") and from the *current*
d ("fresh").  Here we ask whether a single Poincare embedding of the initial
D gives strictly better static candidate sets than D itself — if so, a
sparse-candidate NJ could use embedding-space neighbours instead of raw
distance neighbours.

Protocol mirrors nj_trace's static scheme:
  - leaf candidate list = top-K neighbours by G (geodesic) vs by D
  - per NJ round, cherry (i,j) over clusters; rank = min over leaves
    l in i, l' in j of pos(l' in cand(l)) and symmetric
  - Recall@k = fraction of rounds with rank < k

Usage: hyp_nj_cand.py INPUT.fasta [--kmer K] [--dim D] [--iters N]
"""
import argparse
import sys
import numpy as np

from hyp_embed_test import embed, pdist, EPS  # reuse verified RSGD + metrics


def read_fasta(path):
    seqs = []
    cur = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur).upper())
                cur = []
        else:
            cur.append(line)
    if cur:
        seqs.append("".join(cur).upper())
    return seqs


def kmer_D(seqs, k=5):
    """D = 1 - |ki ∩ kj| / min(|ki|, |kj|) — same metric as msa_ref.cpp."""
    enc = {"T": 0, "C": 1, "A": 2, "G": 3, "U": 0}
    sets = []
    for s in seqs:
        km = set()
        h, run = 0, 0
        for ch in s:
            b = enc.get(ch, -1)
            if b < 0:
                h, run = 0, 0
                continue
            h = (h * 4 + b) % (4 ** k)
            run += 1
            if run >= k:
                km.add(h)
        sets.append(km)
    n = len(sets)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(sets[i] & sets[j])
            mn = min(len(sets[i]), len(sets[j]))
            D[i, j] = D[j, i] = 1.0 - inter / mn if mn else 1.0
    return D


def top_k_lists(M, K):
    """Per-row K nearest others by M, ascending (dist, idx) — like nj_trace."""
    n = M.shape[0]
    idx = np.argsort(M + np.eye(n) * 1e18, axis=1)[:, :K]
    return idx


def nj_round_ranks(d, cand_lists, K):
    """Run NJ on d; per round record cherry rank in the STATIC cand_lists
    (cluster candidates = union of its leaves' lists, position = first leaf
    hitting a leaf of the other cluster — mirrors nj_trace rank_static)."""
    n = d.shape[0]
    NN = 2 * n - 1
    D = np.zeros((NN, NN))
    D[:n, :n] = d
    leaves_in = [[i] for i in range(n)]
    alive = list(range(n))
    pos = {}                     # leaf -> rank lookup per round is expensive;
    ranks = []
    while len(alive) > 2:
        m = len(alive)
        a = np.ix_(alive, alive)
        dm = D[a]
        r = dm.sum(axis=1)
        Q = (m - 2) * dm - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        bi, bj = np.unravel_index(np.argmin(Q), Q.shape)
        xi, xj = alive[bi], alive[bj]

        # rank: for every leaf l in cluster i, earliest position in cand[l]
        # where a leaf of cluster j appears; symmetric; min.
        cl_i = set(leaves_in[xi])
        cl_j = set(leaves_in[xj])
        best = K
        for l in leaves_in[xi]:
            lst = cand_lists[l]
            for p in range(min(best, K)):
                if lst[p] in cl_j:
                    best = p
                    break
        for l in leaves_in[xj]:
            lst = cand_lists[l]
            for p in range(min(best, K)):
                if lst[p] in cl_i:
                    best = p
                    break
        ranks.append(best)

        u = len(leaves_in)
        leaves_in.append(leaves_in[xi] + leaves_in[xj])
        rest = [x for k2, x in enumerate(alive) if k2 != bi and k2 != bj]
        for v in rest:
            D[u, v] = D[v, u] = 0.5 * (D[xi, v] + D[xj, v] - D[xi, xj])
        alive = rest + [u]
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta")
    ap.add_argument("--kmer", type=int, default=5)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--kmax", type=int, default=128)
    args = ap.parse_args()

    seqs = read_fasta(args.fasta)
    n = len(seqs)
    print(f"{args.fasta}: n={n}", file=sys.stderr)
    D = kmer_D(seqs, args.kmer)

    cand_D = top_k_lists(D, args.kmax)
    G, stress = embed(D, args.dim, args.iters)
    cand_G = top_k_lists(G, args.kmax)
    print(f"embedding dim={args.dim} rel-stress={stress:.4f}", file=sys.stderr)

    rD = nj_round_ranks(D, cand_D, args.kmax)
    rG = nj_round_ranks(D, cand_G, args.kmax)

    print("cand\t" + "\t".join(f"recall@{k}" for k in (8, 16, 32, 64, 128)
                              if k <= args.kmax))
    for tag, rr in (("rawD", rD), ("hypG", rG)):
        row = [f"{sum(r < k for r in rr) / len(rr):.4f}"
               for k in (8, 16, 32, 64, 128) if k <= args.kmax]
        print(tag + "\t" + "\t".join(row))


if __name__ == "__main__":
    main()
