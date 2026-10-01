#!/usr/bin/env python3
"""Experiment H1: does a Poincare embedding of patristic distances +
NJ decode recover a known simulated tree?

Pipeline: true tree -> patristic D -> RSGD embedding in Poincare ball
-> geodesic matrix G -> NJ(G) -> RF vs true tree. Control: NJ(D).

Pure numpy (no geoopt dependency). n<=~500 keeps O(n^2) pair loops fine.

Usage: hyp_embed_test.py TRUE_TREE.nwk [n_taxa_limit] [dim] [iters]
"""
import sys
import numpy as np
from io import StringIO
# Bio is only needed by __main__ (tree read / RF / NJ decode); imported lazily
# there so other scripts can reuse embed/pdist without the dependency.

SEED = 7
rng = np.random.default_rng(SEED)

# ---------- tree -> patristic distances ---------------------------------
def patristic(tree, names):
    n = len(names)
    idx = {t.name: i for i, t in enumerate(tree.get_terminals())}
    D = np.zeros((n, n))
    terms = tree.get_terminals()
    for i in range(n):
        for j in range(i + 1, n):
            d = tree.distance(terms[idx[names[i]]], terms[idx[names[j]]])
            D[i, j] = D[j, i] = d
    return D

# ---------- Poincare ball ----------------------------------------------
EPS = 1e-5

def pdist(X):
    """Pairwise Poincare ball distances, X (n,d) -> (n,n)."""
    n = X.shape[0]
    sq = np.sum(X * X, axis=1)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a = 1.0 - sq[i]
            b = 1.0 - sq[j]
            u = 1.0 + 2.0 * np.sum((X[i] - X[j]) ** 2) / (a * b)
            G[i, j] = G[j, i] = np.arccosh(max(u, 1.0 + EPS))
    return G

def pdist_grad(x, y):
    """d d_H(x,y)/dx. Derived: (4/(b*sqrt(u^2-1))) * (x*(||y||^2-2<x,y>+1)/a^2 - y/a).
    Verified against finite differences (the commonly quoted NK2017
    arrangement has x/y roles swapped, giving -grad)."""
    a = 1.0 - np.sum(x * x)
    b = 1.0 - np.sum(y * y)
    u = 1.0 + 2.0 * np.sum((x - y) ** 2) / (a * b)
    u = max(u, 1.0 + EPS)
    c = (np.sum(y * y) - 2.0 * np.dot(x, y) + 1.0) / (a * a)
    return 4.0 / (b * np.sqrt(u * u - 1.0)) * (c * x - y / a)

def embed(D, dim, iters, lr=0.1, scale=None):
    n = D.shape[0]
    if scale is None:
        scale = D.max() / 2.0           # push distances toward boundary
    Dn = D / scale
    X = rng.uniform(-0.01, 0.01, (n, dim))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for it in range(iters):
        i, j = pairs[rng.integers(len(pairs))]
        d_h = np.arccosh(max(
            1.0 + 2.0 * np.sum((X[i] - X[j]) ** 2) /
            ((1 - np.sum(X[i] ** 2)) * (1 - np.sum(X[j] ** 2))),
            1.0 + EPS))
        e = 2.0 * (d_h - Dn[i, j])
        lr_t = lr / (1.0 + 4.0 * it / iters)   # decay
        for k, l in ((i, j), (j, i)):
            g = e * pdist_grad(X[k], X[l])
            gn = np.linalg.norm(g)
            if gn > 1.0:
                g = g / gn                      # clip to unit ball
            step = ((1.0 - np.sum(X[k] ** 2)) ** 2 / 4.0) * g
            X[k] = X[k] - lr_t * step
            nm = np.linalg.norm(X[k])
            if nm > 1.0 - 1e-4:
                X[k] *= (1.0 - 1e-4) / nm
    G = pdist(X) * scale
    stress = np.sqrt(np.mean((pdist(X) - Dn) ** 2)) / np.mean(Dn)
    return G, stress

# ---------- RF distance --------------------------------------------------
def splits(tree, all_names):
    n = len(all_names)
    pos = {nm: i for i, nm in enumerate(all_names)}
    out = set()
    for cl in tree.get_nonterminals():
        bits = frozenset(pos[t.name] for t in cl.get_terminals())
        comp = frozenset(range(n)) - bits
        out.add(min(bits, comp, key=len))
    out.discard(frozenset())
    return out

def rf(t1, t2, names):
    s1, s2 = splits(t1, names), splits(t2, names)
    return len(s1 - s2) + len(s2 - s1)

def nj_tree(D, names):
    dm = DistanceMatrix(names,
                        [list(D[i, :i + 1]) for i in range(len(names))])
    return DistanceTreeConstructor().nj(dm)

if __name__ == "__main__":
    from Bio import Phylo
    from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
    tree_path = sys.argv[1]
    dim = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 200000
    true = Phylo.read(tree_path, "newick")
    for c in true.find_clades():
        if c.branch_length is None:
            c.branch_length = 1.0   # unit edges -> topological distance
    names = [t.name for t in true.get_terminals()]
    n = len(names)
    print(f"true tree: {n} tips", file=sys.stderr)
    D = patristic(true, names)

    t_nj = nj_tree(D, names)
    print(f"NJ(D)  RF vs true: {rf(t_nj, true, names)}")

    G, stress = embed(D, dim, iters)
    print(f"embedding dim={dim} rel-stress={stress:.4f}")
    t_h = nj_tree(G, names)
    print(f"NJ(G)  RF vs true: {rf(t_h, true, names)}")
    print(f"RF(NJ(G), NJ(D)) = {rf(t_h, t_nj, names)}")
