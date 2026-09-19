#!/usr/bin/env python3
"""Robinson-Foulds distance between two Newick trees (unweighted, splits).

Usage: rf_distance.py a.tre b.tre
Trees need not share all tips: the comparison is restricted to the shared
taxon set (both trees pruned implicitly by intersecting split taxa).
"""
import re
import sys


def strip_square_annotations(s):
    """Remove nested NHX/BEAST square-bracket annotations."""
    for _ in range(10):
        cleaned = re.sub(r"\[[^\[\]]*\]", "", s)
        if cleaned == s:
            break
        s = cleaned
    return s


def parse_newick(s):
    """Minimal Newick parser -> (children lists, labels)."""
    s = strip_square_annotations(s).strip().rstrip(";")
    children = {}
    labels = {}
    nid = 0
    stack = []
    i = 0
    n = len(s)
    # virtual root: top-level tokens (unrooted trees may start with a leaf)
    nid += 1
    children[nid] = []
    cur = nid
    while i < n:
        c = s[i]
        if c == "(":
            nid += 1
            children[nid] = []
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
            cur = stack.pop() if stack else cur
        else:
            j = i
            while j < n and s[j] not in ",()":
                j += 1
            lab = s[i:j].split(":")[0].strip().strip("'\"")
            nid += 1
            children[nid] = []
            if lab:
                labels[nid] = lab
            children[cur].append(nid)
            i = j
    return children, labels, nid


def read_newick(path):
    with open(path) as handle:
        return parse_newick(handle.read())


def splits(tree_path):
    children, labels, _ = read_newick(tree_path)
    # leaf labels only (internal labels like bootstrap supports excluded)
    taxa = sorted({labels[u] for u in children
                   if not children[u] and u in labels})
    idx = {t: k for k, t in enumerate(taxa)}
    allm = (1 << len(taxa)) - 1
    out = set()
    # pre-order traversal; reversed gives children before parents
    root = min(children)
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
    mask = {}
    for u in reversed(order):
        if not children[u]:
            mask[u] = (1 << idx[labels[u]]) if labels.get(u) in idx else 0
        else:
            m = 0
            for v in children[u]:
                m |= mask[v]
            mask[u] = m
    for u in order:
        if not children[u]:
            continue
        m = mask[u]
        other = allm ^ m
        # nontrivial split: both sides need >= 2 taxa; min() canonicalizes,
        # it does not pick the smaller side
        if (m & (m - 1)) and (other & (other - 1)):
            out.add(min(m, other))
    return out, taxa


# restrict both to shared taxa: drop splits whose taxa differ? Proper RF on
# shared set requires pruning; here we just report on each tree's splits
# over the union taxa indexing -- safer: rebuild with union labels.
# Simpler correct approach: compare split sets built on each tree's OWN
# taxon list, projecting onto shared taxa.
def project(tree_path, keep):
    children, labels, _ = read_newick(tree_path)
    keep = set(keep)
    root = min(children)
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
    mask = {}
    for u in reversed(order):
        if not children[u]:
            t = labels.get(u)
            mask[u] = (1 << idx[t]) if t in keep else 0
        else:
            m = 0
            for v in children[u]:
                m |= mask[v]
            mask[u] = m
        m = mask[u]
        if not children[u]:
            continue
        other = allm ^ m
        if (m & (m - 1)) and (other & (other - 1)):
            out.add(min(m, other))
    return out, taxa


def rf_distance(path_a, path_b):
    """Return (rf, rf_norm, n_shared) on the shared taxon set."""
    _, ta = splits(path_a)
    _, tb = splits(path_b)
    shared = sorted(set(ta) & set(tb))
    sa, _ = project(path_a, shared)
    sb, _ = project(path_b, shared)
    rf = len(sa ^ sb)
    rf_norm = rf / (len(sa) + len(sb)) if (sa or sb) else 0.0
    return rf, rf_norm, len(shared)


if __name__ == '__main__':
    rf, rf_norm, nshared = rf_distance(sys.argv[1], sys.argv[2])
    print(f"shared_taxa={nshared} "
          f"RF={rf} RF_norm={rf_norm:.4f}")
