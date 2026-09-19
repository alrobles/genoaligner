#!/usr/bin/env python3
# SP (sum-of-pairs, a.k.a. Q) and TC (total-column) scores of a test MSA
# against a reference alignment, qscore-compatible semantics:
#
#   - core columns = reference columns containing >=1 UPPERCASE residue
#     (drive5 bench1.0 convention; lower-case regions are unscored)
#   - SP: over all core columns, fraction of reference-aligned residue
#     pairs that are also aligned together in the test MSA
#   - TC: fraction of core columns exactly reproduced -- a test column
#     must contain the same residue-bearing row set (no more, no fewer)
#   - reference sequences missing from the test are skipped
#     (qscore -ignoremissingseqs); test letter case is ignored
#
#   msa_bench_score.py test.fasta ref.fasta            -> per-file scores
#   msa_bench_score.py --scan TESTDIR REFDIR           -> per-set table
#
import os
import sys


def read_fasta(path):
    rows = []
    name, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    rows.append((name, "".join(buf)))
                name = line[1:].split()[0]
                buf = []
            elif line.strip():
                buf.append(line.strip())
    if name is not None:
        rows.append((name, "".join(buf)))
    return rows


def is_gap(ch):
    return not ch.isalpha()


def score(test_rows, ref_rows):
    # ref_rows: list of (id, aligned string); core = column has uppercase
    ref_seqs = [r for _, r in ref_rows]
    L = len(ref_seqs[0])
    assert all(len(r) == L for r in ref_seqs), "ragged reference"
    core = [c for c in range(L)
            if any(r[c].isupper() for r in ref_seqs)]

    # test row lookup by id
    test_by_id = {i: r for i, r in test_rows}

    # For every ref row present in test: residue ordinal -> test column
    present = []
    n_missing = 0
    for rid, r in ref_rows:
        t = test_by_id.get(rid)
        if t is None:
            n_missing += 1
            continue
        # map each residue ordinal to its test column
        pos = []
        for c, ch in enumerate(t):
            if not is_gap(ch):
                pos.append(c)
        present.append((rid, r, t, pos))

    # test column -> set of row indices (into `present`) with a residue
    ncol_t = max((len(t) for _, _, t, _ in present), default=0)
    col_rows = [set() for _ in range(ncol_t)]
    for k, (_, _, t, _) in enumerate(present):
        for c, ch in enumerate(t):
            if not is_gap(ch):
                col_rows[c].add(k)

    pairs_ok = pairs_tot = 0
    cols_ok = 0
    for c in core:
        # ref rows (indices into `present`) bearing a residue at column c
        S = [k for k, (_, r, _, _) in enumerate(present) if not is_gap(r[c])]
        # ordinal index of that residue within each row's ungapped string
        residx = {}
        for k in S:
            residx[k] = sum(1 for x in present[k][1][:c] if not is_gap(x))
        npairs = len(S) * (len(S) - 1) // 2
        pairs_tot += npairs
        if not S:
            continue
        k0 = S[0]
        T = present[k0][3][residx[k0]]
        ok_pairs = True
        for j in S[1:]:
            if present[j][3][residx[j]] != T:
                ok_pairs = False
                break
        # TC: test column T must contain exactly the row set S
        if ok_pairs and col_rows[T] == set(S):
            cols_ok += 1
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                i, j = S[a], S[b]
                if present[i][3][residx[i]] == present[j][3][residx[j]]:
                    pairs_ok += 1
    sp = pairs_ok / pairs_tot if pairs_tot else 0.0
    tc = cols_ok / len(core) if core else 0.0
    return sp, tc, pairs_tot, len(core), n_missing


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--scan":
        tdir, rdir = sys.argv[2], sys.argv[3]
        names = sorted(os.listdir(rdir))
        tot_sp = tot_tc = 0.0
        npair_t = ncol_t = 0
        pok = cok = 0
        print("set\tSP\tTC\tpairs\tcols\tmissing")
        for nm in names:
            tp = os.path.join(tdir, nm)
            rp = os.path.join(rdir, nm)
            if not os.path.exists(tp):
                print(f"{nm}\tMISSING", file=sys.stderr)
                continue
            sp, tc, np, nc, miss = score(read_fasta(tp), read_fasta(rp))
            print(f"{nm}\t{sp:.4f}\t{tc:.4f}\t{np}\t{nc}\t{miss}")
            tot_sp += sp * np
            tot_tc += tc * nc
            npair_t += np
            ncol_t += nc
        wsp = tot_sp / npair_t if npair_t else 0.0
        wtc = tot_tc / ncol_t if ncol_t else 0.0
        print(f"ALL\t{wsp:.4f}\t{wtc:.4f}\t{npair_t}\t{ncol_t}\t-",
              file=sys.stdout)
    elif len(sys.argv) == 3:
        sp, tc, np, nc, miss = score(read_fasta(sys.argv[1]),
                                     read_fasta(sys.argv[2]))
        print(f"SP={sp:.4f} TC={tc:.4f} pairs={np} cols={nc} missing={miss}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
