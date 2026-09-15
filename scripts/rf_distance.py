#!/usr/bin/env python3
"""Robinson-Foulds distance between two Newick trees (unweighted, splits).

Usage: rf_distance.py a.tre b.tre
Trees need not share all tips: the comparison is restricted to the shared
taxon set (both trees pruned implicitly by intersecting split taxa).
"""
import sys


def parse_newick(s):
    """Minimal Newick parser -> (children lists, labels)."""
    s = s.strip().rstrip(";")
    children = {}
    labels = {}
    nid = 0
    stack = []
    i = 0
    n = len(s)
    cur = -1
    while i < n:
        c = s[i]
        if c == "(":
            nid += 1
            children[nid] = []
            if cur >= 0:
                children[cur].append(nid)
            stack.append(cur)
            cur = nid
            i += 1
        elif c == ",":
            i += 1
        elif c == ")":
            i += 1
            # optional internal label / branch length
            j = i
            while j < n and s[j] not in ",)":
                j += 1
            if ":" in s[i:j]:
                lab = s[i:j].split(":")[0].strip()
                if lab:
                    labels[cur] = lab
            i = j
            parent = stack.pop()
            cur = parent
        else:
            j = i
            while j < n and s[j] not in ",()":
                j += 1
            lab = s[i:j].split(":")[0].strip().strip("'\"")
            nid += 1
            labels[nid] = lab
            children[nid] = []
            children[cur].append(nid)
            i = j
    return children, labels, nid


def splits(tree_path):
    children, labels, _ = parse_newick(open(tree_path).read())
    # leaf set under each internal node -> canonical split
    taxa = sorted({v for v in labels.values()
                   if v and not isinstance(v, int)})
    idx = {t: k for k, t in enumerate(taxa)}
    allm = (1 << len(taxa)) - 1
    out = set()
    # post-order
    order = []
    root = max(children)
    stack = [root]
    seen = set()
    while stack:
        u = stack[-1]
        if u in seen:
            stack.pop()
            continue
        seen.add(u)
        order.append(u)
        stack.extend(children[u])
    mask = {}
    for u in order:
        if not children[u]:
            mask[u] = 1 << idx[labels[u]]
        else:
            m = 0
            for v in children[u]:
                m |= mask[v]
            mask[u] = m
    for u, m in mask.items():
        if not children[u]:
            continue
        c = min(m, allm ^ m)
        if c and (c & (c - 1)):        # exclude trivial singletons
            out.add(c)
    return out, taxa


a, ta = splits(sys.argv[1])
b, tb = splits(sys.argv[2])
# restrict both to shared taxa: drop splits whose taxa differ? Proper RF on
# shared set requires pruning; here we just report on each tree's splits
# over the union taxa indexing -- safer: rebuild with union labels.
# Simpler correct approach: compare split sets built on each tree's OWN
# taxon list, projecting onto shared taxa.
def project(tree_path, keep):
    children, labels, _ = parse_newick(open(tree_path).read())
    keep = set(keep)
    mask = {}
    root = max(children)
    order = []
    stack = [root]
    seen = set()
    while stack:
        u = stack[-1]
        if u in seen:
            stack.pop()
            continue
        seen.add(u)
        order.append(u)
        stack.extend(children[u])
    taxa = sorted(keep)
    idx = {t: i for i, t in enumerate(taxa)}
    allm = (1 << len(taxa)) - 1
    out = set()
    for u in order:
        if not children[u]:
            t = labels[u]
            mask[u] = (1 << idx[t]) if t in keep else 0
        else:
            m = 0
            for v in children[u]:
                m |= mask[v]
            mask[u] = m
        m = mask[u]
        if not children[u] or m == 0 or m == allm:
            continue
        c = min(m, allm ^ m)
        if c and (c & (c - 1)):
            out.add(c)
    return out, taxa


shared = sorted(set(ta) & set(tb))
sa, _ = project(sys.argv[1], shared)
sb, _ = project(sys.argv[2], shared)
rf = len(sa ^ sb)
rf_norm = rf / (len(sa) + len(sb)) if (sa or sb) else 0.0
print(f"shared_taxa={len(shared)} splits_a={len(sa)} splits_b={len(sb)} "
      f"RF={rf} RF_norm={rf_norm:.4f}")
