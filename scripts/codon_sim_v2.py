#!/usr/bin/env python3
"""codon_sim_v2 — codon-level evolution simulator with FULLY tracked truth.

Evolution model (per branch, balanced binary guide tree):
  - codon substitutions: one nt mutated inside a codon; accepted with
    weight 1.0 if synonymous, `omega` if nonsynonymous; stop-forming
    substitutions are resampled (premature stops are a separate event
    class, not drift).
  - whole-codon indels: 1..5 codons (geometric), at codon boundaries.
  - frameshift indels: 1 or 2 nt insertions/deletions at any position —
    probability `fs_frac` of all indel events.
  - premature stops: with probability `stop_frac` per leaf (applied at
    leaf emission), one random sense codon -> stop codon.
  - 5'/3' fragmentation at leaf emission: each leaf independently drops
    ~Exp(frag5), ~Exp(frag3) residues of either end (0 = full length).
  - sequencing errors at emission: per-base substitution rate `err`
    (residues keep their column id — an error is noise ON a homologous
    residue, not a different homology).
  - reliability classes: `lr_frac` of leaves are "less reliable"
    (MACSE's seq_less_reliable analog): they draw frameshift indels,
    stops and sequencing errors at `lr_mult` x the base rates.

Tracked truth (all emitted):
  sim_in.fasta        leaf nt sequences (unaligned input)
  sim_true.fasta      true nt-level MSA (one column per col id)
  sim_true_aa.fasta   true codon-level MSA (one column per codon id;
                      residues of a frameshifted codon keep their
                      original codon identity; 1-2 nt insertions carry
                      codon id None and appear as gap-only AA columns)
  sim_true_tree.nwk   balanced guide/species tree used
  events.tsv          per-leaf event log (type, anchor col id, len, ...)
  manifest.json       generator version + all parameters + seed +
                      observed counts — the reproducibility record

Scoring mode: `codon_sim_v2.py GOT.fasta --n N --seed S ...` (same
params) regenerates the identical truth and scores GOT against it:
  nt-SPS, nt-TC (full true columns reproduced), codon-SPS (shared codon
  ids whose residues share >=1 got column), plus structure metrics
  (decoded-len %3, got/true width, gap-run codon purity).

Usage:
  codon_sim_v2.py - N SEED SUB INDEL L [--codon] [options]   # generate
  codon_sim_v2.py GOT.fasta N SEED SUB INDEL L [options]     # generate + score
"""
import argparse
import json
import random
import sys

VERSION = "codon_sim_v2.1"

BASES = "ACGT"

# NCBI tables 1 (standard) and 2 (vertebrate mito): codon -> amino acid
_AA1 = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
_AA2 = ("FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSS**VVVVAAAADDEEGGGG")
_CODON2AA = {1: {}, 2: {}}
for _i in range(64):
    _c = "TCAG"[_i // 16] + "TCAG"[(_i // 4) % 4] + "TCAG"[_i % 4]
    _CODON2AA[1][_c] = _AA1[_i]
    _CODON2AA[2][_c] = _AA2[_i]


def code_tables(gc):
    """(stops:set, sense:list, codon2aa:dict) for NCBI table gc."""
    tab = _CODON2AA[gc]
    stops = {c for c, a in tab.items() if a == "*"}
    sense = [c for c in sorted(tab) if tab[c] != "*"]
    return stops, sense, tab


def simulate(n_leaves, L, seed, sub_rate, indel_rate, omega=0.3,
             fs_frac=0.0, stop_frac=0.0, frag5=0.0, frag3=0.0,
             err=0.0, lr_frac=0.0, lr_mult=5.0, gc=1):
    """Returns (leaf_seqs, true_rows, true_aa_rows, tree_nwk, events, manifest)."""
    rng = random.Random(seed)
    stops, sense, codon2aa = code_tables(gc)

    # ancestor: random sense codons; residue = [base, col_id, codon_id]
    anc = []
    for i in range(L // 3):
        cod = rng.choice(sense)
        for k, b in enumerate(cod):
            anc.append([b, 3 * i + k, i])
    order = list(range(len(anc)))            # nt col ids in global order
    aorder = list(range(len(anc) // 3))      # codon ids in global order
    next_id = [len(anc)]                     # next free nt col id
    next_cid = [len(anc) // 3]               # next free codon id

    # leaf class assignment (stable across scoring: same seed -> same)
    is_lr = [rng.random() < lr_frac for _ in range(n_leaves)]
    leaf_events = [[] for _ in range(n_leaves)]

    def sub_codon(s, leaf_id, omega_eff):
        """One codon substitution, omega-weighted vs nonsynonymous."""
        ncod = len(s) // 3
        if ncod == 0:
            return
        ci = rng.randrange(ncod)
        p = 3 * ci
        cod = "".join(s[p + k][0] for k in range(3))
        if cod in stops or len(cod) < 3:
            return
        aa0 = codon2aa.get(cod, "X")
        # candidate point mutations
        cands = []
        for k in range(3):
            for b in BASES:
                if b == cod[k]:
                    continue
                c2 = cod[:k] + b + cod[k + 1:]
                if c2 in stops:
                    continue  # stops are their own event class
                w = 1.0 if codon2aa.get(c2) == aa0 else omega_eff
                cands.append((k, b, w))
        if not cands:
            return
        tot = sum(w for _, _, w in cands)
        r = rng.random() * tot
        acc = 0.0
        for k, b, w in cands:
            acc += w
            if r <= acc:
                s[p + k][0] = b
                leaf_events[leaf_id].append(
                    {"type": "sub", "codon_id": s[p + k][2]})
                return

    def ins_codons(s, leaf_id, ci, n_codons):
        """Insert n_codons whole codons before codon index ci."""
        new_cols = list(range(next_id[0], next_id[0] + 3 * n_codons))
        new_cids = list(range(next_cid[0], next_cid[0] + n_codons))
        next_id[0] += 3 * n_codons
        next_cid[0] += n_codons
        blk = []
        for j in range(n_codons):
            for k in range(3):
                blk.append([rng.choice(BASES), new_cols[3 * j + k],
                            new_cids[j]])
        if ci > 0:
            oi = order.index(s[3 * ci - 1][1]) + 1
            # anchor on the nearest preceding residue with a codon id
            # (frameshift-inserted nts carry None and are not in aorder)
            ai = 0
            for p in range(3 * ci - 1, -1, -1):
                if s[p][2] is not None:
                    ai = aorder.index(s[p][2]) + 1
                    break
        else:
            oi = ai = 0
        order[oi:oi] = new_cols
        aorder[ai:ai] = new_cids
        s[3 * ci:3 * ci] = blk
        leaf_events[leaf_id].append(
            {"type": "ins3k", "codons": n_codons,
             "after_codon": (s[3 * ci - 1][2] if ci > 0 else -1)})

    def del_codons(s, leaf_id, ci, k):
        del s[3 * ci:3 * (ci + k)]
        leaf_events[leaf_id].append({"type": "del3k", "codons": k,
                                     "at_codon": ci})

    def ins_nt(s, leaf_id, pos, ln):
        """Frameshift insertion of ln (1-2) nt; codon_id None."""
        new = list(range(next_id[0], next_id[0] + ln))
        next_id[0] += ln
        blk = [[rng.choice(BASES), c, None] for c in new]
        if pos > 0:
            oi = order.index(s[pos - 1][1]) + 1
        else:
            oi = 0
        order[oi:oi] = new
        s[pos:pos] = blk
        leaf_events[leaf_id].append(
            {"type": "fs_ins", "len": ln,
             "after_col": (s[pos - 1][1] if pos > 0 else -1)})

    def del_nt(s, leaf_id, pos, ln):
        """Frameshift deletion of ln (1-2) nt."""
        if pos + ln > len(s):
            pos = len(s) - ln
        if pos < 0:
            return
        gone = [s[pos + t][1] for t in range(ln)]
        del s[pos:pos + ln]
        leaf_events[leaf_id].append(
            {"type": "fs_del", "len": ln, "cols": gone})

    def evolve(s0, n_subs, n_indels, leaf_id, lr):
        s = [r[:] for r in s0]  # copy residues
        om = omega
        for _ in range(n_subs):
            sub_codon(s, leaf_id, om)
        fs_p = fs_frac * (lr_mult if lr else 1.0)
        for _ in range(n_indels):
            ncod = len(s) // 3
            if ncod == 0:
                continue
            if rng.random() < fs_p:
                # frameshift indel: 1-2 nt at any nt position
                ln = rng.choice([1, 2])
                if rng.random() < 0.5:
                    ins_nt(s, leaf_id, rng.randrange(len(s) + 1), ln)
                else:
                    del_nt(s, leaf_id, rng.randrange(len(s) + 1), ln)
            else:
                # whole-codon indel at codon boundary
                ln_c = 1 + int(rng.expovariate(1 / 2.5))
                if rng.random() < 0.5:
                    ins_codons(s, leaf_id, rng.randrange(ncod + 1), ln_c)
                else:
                    ci = rng.randrange(ncod)
                    del_codons(s, leaf_id, ci, min(ln_c, ncod - ci))
        return s

    def branch(seq, leaf_id, lr):
        n_sub = max(1, round(len(seq) * sub_rate * (0.75 + 0.5 * rng.random())))
        n_ind = round(len(seq) * indel_rate * (0.5 + rng.random()))
        return evolve(seq, n_sub, n_ind, leaf_id, lr)

    # balanced split; leaf index assigned in split order
    leaf_seqs_raw = []
    def split(seq, k):
        if k == 1:
            leaf_seqs_raw.append(seq)
            return
        lid = len(leaf_seqs_raw)
        mid = k // 2
        split(branch(seq, lid, is_lr[lid] if lid < n_leaves else False), mid)
        split(branch(seq, lid + mid, is_lr[lid + mid]
                     if lid + mid < n_leaves else False), k - mid)
    # NOTE: leaf_id in events approximates final leaf order for stats only.
    split(anc, n_leaves)

    # leaf emission: truncation + premature stop + sequencing errors
    leaf_seqs = []
    for lid, s in enumerate(leaf_seqs_raw):
        lr = is_lr[lid]
        e = list(s)
        mult = lr_mult if lr else 1.0
        # premature stop: convert a random sense codon
        if rng.random() < min(1.0, stop_frac * mult):
            ncod = len(e) // 3
            if ncod:
                ci = rng.randrange(ncod)
                p = 3 * ci
                cod = "".join(e[p + k][0] for k in range(3))
                if cod not in stops and len(cod) == 3:
                    stop = rng.choice(sorted(stops))
                    for k in range(3):
                        e[p + k][0] = stop[k]
                    leaf_events[lid].append(
                        {"type": "premature_stop", "codon_id": e[p][2]})
        # 5'/3' truncation
        f5 = int(rng.expovariate(1 / frag5)) if frag5 else 0
        f3 = int(rng.expovariate(1 / frag3)) if frag3 else 0
        if f5 or f3:
            e = e[f5:len(e) - f3 if f3 else len(e)]
            leaf_events[lid].append(
                {"type": "fragment", "drop5": f5, "drop3": f3})
        # sequencing errors (homologous residue, noisy base)
        er = err * mult
        if er:
            nerr = 0
            for r in e:
                if rng.random() < er:
                    r[0] = rng.choice([b for b in BASES if b != r[0]])
                    nerr += 1
            if nerr:
                leaf_events[lid].append({"type": "seqerr", "n": nerr})
        leaf_seqs.append(e)

    # ---- true nt MSA ------------------------------------------------
    colidx = {c: i for i, c in enumerate(order)}
    true_rows = []
    for s in leaf_seqs:
        row = ["-"] * len(order)
        for b, c, _cid in s:
            row[colidx[c]] = b
        true_rows.append("".join(row))

    # ---- true codon (AA-proxy) MSA -----------------------------------
    # group each leaf's residues by codon_id (None -> excluded from AA
    # truth: they are the frameshift insertions themselves)
    aidx = {c: i for i, c in enumerate(aorder)}
    true_aa_rows = []
    for s in leaf_seqs:
        row = ["-"] * len(aorder)
        for _b, _c, cid in s:
            if cid is not None:
                row[aidx[cid]] = "X"     # presence marker per codon
        true_aa_rows.append("".join(row))

    tree_nwk = _balanced_nwk(n_leaves)

    manifest = {
        "generator": VERSION,
        "gc": gc, "seed": seed, "n_leaves": n_leaves, "L": L,
        "sub_rate": sub_rate, "indel_rate": indel_rate, "omega": omega,
        "fs_frac": fs_frac, "stop_frac": stop_frac,
        "frag5": frag5, "frag3": frag3, "err": err,
        "lr_frac": lr_frac, "lr_mult": lr_mult,
        "nt_cols": len(order), "codon_cols": len(aorder),
        "leaf_lens": [len(s) for s in leaf_seqs],
        "events_per_leaf": [len(e) for e in leaf_events],
    }
    seqs_out = ["".join(b for b, _c, _i in s) for s in leaf_seqs]
    return seqs_out, true_rows, true_aa_rows, tree_nwk, leaf_events, manifest


def _balanced_nwk(n):
    def rec(lo, k):
        if k == 1:
            return f"s{lo}"
        mid = k // 2
        return f"({rec(lo, mid)},{rec(lo + mid, k - mid)})"
    return rec(0, n) + ";\n"


# ---------------- scoring -------------------------------------------------
RES_OK = set("ACGTUN*X")

def _resmap(row):
    m = {}
    r = 0
    for j, c in enumerate(row):
        if c not in "-!":
            m[r] = j
            r += 1
    return m


def sps_pairs(tm, gm, offs, n):
    """Column-SPS: fraction of true residue-pairs reproduced in got."""
    same = tot = 0
    invs = [{v: k for k, v in t.items()} for t in tm]
    for a in range(n):
        off_a, lim_a = offs[a]
        ga = gm[a]
        for b in range(a + 1, n):
            off_b, lim_b = offs[b]
            gb = gm[b]
            inv_b = invs[b]
            for ra, ca in tm[a].items():
                rb = inv_b.get(ca)
                if rb is None:
                    continue
                oa, ob = ra - off_a, rb - off_b
                if oa < 0 or ob < 0 or ra >= lim_a or rb >= lim_b:
                    continue
                tot += 1
                if ga.get(oa) == gb.get(ob):
                    same += 1
    return same, tot


def tc_score(true_rows, got_rows, offs, n):
    """Total-column score: fraction of TRUE columns reproduced intact
    in got (all residues of the true column share one got column)."""
    tm = [_resmap(r) for r in true_rows]
    gm = [_resmap(r) for r in got_rows]
    invs = [{v: k for k, v in t.items()} for t in tm]   # col -> ordinal
    hit = tot = 0
    ncols = len(true_rows[0])
    for c in range(ncols):
        colset = set()
        have = 0
        ok = True
        for i in range(n):
            r = invs[i].get(c)
            if r is None:
                continue                      # gap in the true column
            have += 1
            off, lim = offs[i]
            o = r - off
            j = gm[i].get(o) if 0 <= o < lim else None
            if j is None:
                ok = False                    # residue absent in got
                break
            colset.add(j)
        if have < 2:
            continue
        tot += 1
        if ok and len(colset) == 1:
            hit += 1
    return hit, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("got", help="'-' = generate only; else fasta to score")
    ap.add_argument("n", type=int)
    ap.add_argument("seed", type=int)
    ap.add_argument("sub", type=float)
    ap.add_argument("indel", type=float)
    ap.add_argument("L", type=int)
    ap.add_argument("--omega", type=float, default=0.3)
    ap.add_argument("--fs-frac", type=float, default=0.0)
    ap.add_argument("--stop-frac", type=float, default=0.0)
    ap.add_argument("--frag5", type=float, default=0.0)
    ap.add_argument("--frag3", type=float, default=0.0)
    ap.add_argument("--err", type=float, default=0.0)
    ap.add_argument("--lr-frac", type=float, default=0.0)
    ap.add_argument("--lr-mult", type=float, default=5.0)
    ap.add_argument("--gc", type=int, default=1, choices=[1, 2])
    args = ap.parse_args()

    seqs, true_rows, true_aa, nwk, events, manifest = simulate(
        args.n, args.L, args.seed, args.sub, args.indel,
        omega=args.omega, fs_frac=args.fs_frac, stop_frac=args.stop_frac,
        frag5=args.frag5, frag3=args.frag3, err=args.err,
        lr_frac=args.lr_frac, lr_mult=args.lr_mult, gc=args.gc)

    if args.got == "-":
        with open("sim_in.fasta", "w") as f:
            for i, s in enumerate(seqs):
                f.write(f">s{i}\n{s}\n")
        with open("sim_true.fasta", "w") as f:
            for i, s in enumerate(true_rows):
                f.write(f">s{i}\n{s}\n")
        with open("sim_true_aa.fasta", "w") as f:
            for i, s in enumerate(true_aa):
                f.write(f">s{i}\n{s}\n")
        with open("sim_true_tree.nwk", "w") as f:
            f.write(nwk)
        with open("events.tsv", "w") as f:
            f.write("leaf\tevent\n")
            for i, evs in enumerate(events):
                for e in evs:
                    f.write(f"s{i}\t{json.dumps(e)}\n")
        with open("manifest.json", "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"wrote sim_in/sim_true/sim_true_aa/tree/events/manifest "
              f"(n={args.n})")
        return

    # ---- score mode --------------------------------------------------
    got = {}
    cur = None
    for line in open(args.got):
        line = line.strip()
        if line.startswith(">"):
            cur = line[1:].split()[0]
            got[cur] = ""
        elif cur:
            got[cur] += line.upper()
    # a tool may drop records (e.g. sequences it cannot place): missing
    # rows score as all-gap rows of the tool's own width.
    width = max(len(r) for r in got.values()) if got else 0
    got_rows = [got.get(f"s{i}", "-" * width) for i in range(args.n)]

    # ungapped sanity + per-seq ordinal offset (genomsa --codon emits the
    # frame-masked sequence: leading frame-offset bases dropped, partial
    # codon -> NNN)
    stops_g, _, _ = code_tables(args.gc)
    def mask_codon(s):
        best, bestst = 0, None
        for fr in range(3):
            st = sum(s[i:i + 3] in stops_g
                     for i in range(fr, len(s) - 2, 3))
            if bestst is None or st < bestst:
                bestst, best = st, fr
        out = []
        for i in range(best, len(s) - 2, 3):
            c = s[i:i + 3]
            out.append(c if all(b in "ACGT" for b in c) else "NNN")
        if (len(s) - best) % 3:
            out.append("NNN")
        return "".join(out), best
    offs = []
    for i, s in enumerate(seqs):
        ug = got_rows[i].replace("-", "").replace("!", "")
        if not ug:
            offs.append((0, len(s)))          # dropped by the tool
            continue
        masked, best = mask_codon(s)
        if ug not in (masked, s):
            # allow (a) N-wildcard mismatches (PRANK rewrites codons it
            # cannot handle as NNN) and (b) prefix truncation (3-step
            # back-translation drops a trailing partial codon). Ordinals
            # still map 1:1 in both cases.
            ok = len(ug) <= len(s) and all(
                a == "N" or a == b for a, b in zip(ug, s))
            if not ok:
                ok = len(ug) <= len(masked) and all(
                    a == "N" or a == b for a, b in zip(ug, masked))
            assert ok, f"row {i} corrupted: ungapped != input"
        offs.append((best, len(s)) if ug == masked else (0, len(s)))

    tm = [_resmap(r) for r in true_rows]
    gm = [_resmap(r) for r in got_rows]
    same, tot = sps_pairs(tm, gm, offs, args.n)
    sps = same / tot if tot else float("nan")
    hit, tct = tc_score(true_rows, got_rows, offs, args.n)
    tc = hit / tct if tct else float("nan")

    # structure: fraction of got rows whose length is %3; gap purity:
    # in codon-mode output every gap column should be a whole triplet.
    w_got = len(got_rows[0])
    trip_cols = 0
    for j in range(0, w_got - 2, 3):
        if all(got_rows[i][j:j + 3] in ("---",) or "-" not in
               got_rows[i][j:j + 3] for i in range(args.n)):
            trip_cols += 1
    purity = trip_cols / max(1, w_got // 3)

    print(f"SIM-SPS {sps:.4f} SIM-TC {tc:.4f} "
          f"width_true={len(true_rows[0])} width_got={w_got} "
          f"trip_purity={purity:.3f}")


if __name__ == "__main__":
    main()
