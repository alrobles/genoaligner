#!/usr/bin/env python3
"""Build the Fase 3 control set: known identity levels, deterministic.

The masterplan asks for pairs at 50/70/90/100% identity. Two properties matter
more than the exact numbers:

1. DETERMINISM. Fixed seed, so the same set is produced on every machine and
   every run. A parity claim against a set that changes is not a claim.

2. HONEST isD REPORTING. The kernel chases edit distance up to smax (=64 in the
   GPU harness). Pairs whose isD exceeds smax are ABANDONED, not failed -- the
   MI210 run had 61 abandoned cases out of 1007 from its len=256 regimes.
   Abandoned cases must be visible and attributed, never silently dropped, or
   the parity percentage overstates what was actually tested.

So every generated pair carries its isD, and the summary reports how many fall
beyond a given smax.

USAGE
    python3 make_control_set.py --outdir tests/parity/data
    python3 make_control_set.py --outdir /tmp/x --cases 200 --verify-determinism
"""
import argparse
import os
import random
import sys

ALPHABET = "ACGT"

# Identity levels from the masterplan. 100% exercises the zero-score path;
# 50% is the far end of the "phylogenetic regime" the kernel targets.
IDENTITIES = [100, 90, 70, 50]

# Lengths chosen to span the regime while keeping most cases resolvable under
# smax. len=256 at 50% identity will exceed smax=64 by construction, which is
# intentional: it makes the abandoned count non-zero and therefore visible.
LENGTHS = [32, 64, 128, 256]


def mutate(seq: str, ident_pct: int, rng: random.Random) -> str:
    """Introduce substitutions so that roughly ident_pct of positions match."""
    n_mut = int(round(len(seq) * (100 - ident_pct) / 100.0))
    out = list(seq)
    positions = rng.sample(range(len(seq)), min(n_mut, len(seq)))
    for pos in positions:
        alt = [c for c in ALPHABET if c != out[pos]]
        out[pos] = rng.choice(alt)
    return "".join(out)


def add_indel(seq: str, rng: random.Random) -> str:
    """Occasionally drop or insert a base, so gaps are exercised, not just subs."""
    if len(seq) < 8:
        return seq
    if rng.random() < 0.5:
        i = rng.randrange(len(seq))
        return seq[:i] + seq[i + 1:]
    i = rng.randrange(len(seq))
    return seq[:i] + rng.choice(ALPHABET) + seq[i:]


def build(cases: int, seed: int):
    """Return list of (text, pattern, ident, length, label)."""
    rng = random.Random(seed)
    out = []
    for c in range(cases):
        ident = IDENTITIES[c % len(IDENTITIES)]
        length = LENGTHS[(c // len(IDENTITIES)) % len(LENGTHS)]

        text = "".join(rng.choice(ALPHABET) for _ in range(length))
        pattern = mutate(text, ident, rng)
        if c % 7 == 0:
            pattern = add_indel(pattern, rng)

        label = f"len={length} ident={ident}%"
        out.append((text, pattern, ident, length, label))

    # Adversarial edge cases: where off-by-one and sentinel bugs live.
    edge = [
        ("", "", "both empty"),
        ("ACGT", "", "empty pattern"),
        ("", "ACGT", "empty text"),
        ("A", "T", "single mismatch"),
        ("AAAA", "AAAA", "identical"),
        ("AAAA", "AA", "pattern is prefix"),
        ("ACGTACGT", "TTTTTTTT", "no match at all"),
    ]
    for text, pattern, label in edge:
        out.append((text, pattern, -1, len(text), label))
    return out


def edit_distance(a: str, b: str) -> int:
    """Small self-contained DP so this script has no oracle dependency."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j - 1] + (ca != cb), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1]


def summarize(rows, smax: int) -> str:
    """Report the isD distribution and the abandoned count at a given smax."""
    dists = [edit_distance(p, t) for t, p, _, _, _ in rows]
    beyond = sum(1 for d in dists if d > smax)
    n = len(dists)
    return (f"  cases        : {n}\n"
            f"  isD <= {smax:<5} : {n - beyond}  (resolvable)\n"
            f"  isD  > {smax:<5} : {beyond}  (ABANDONED -- must be reported, not dropped)\n"
            f"  max isD      : {max(dists) if dists else 0}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="tests/parity/data")
    ap.add_argument("--cases", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345,
                    help="fixed by default so the set is reproducible")
    ap.add_argument("--smax", type=int, default=64,
                    help="must match the harness's smax for the abandoned count")
    ap.add_argument("--verify-determinism", action="store_true",
                    help="regenerate and confirm byte-identical output")
    args = ap.parse_args()

    rows = build(args.cases, args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"control_seed{args.seed}.tsv")
    with open(path, "w") as fh:
        fh.write("# genoaligner Fase 3 control set\n")
        fh.write(f"# seed={args.seed} cases={args.cases} alphabet={ALPHABET}\n")
        fh.write("# columns: pattern<TAB>text<TAB>ident<TAB>length<TAB>label\n")
        fh.write("pattern\ttext\tident\tlength\tlabel\n")
        for text, pattern, ident, length, label in rows:
            fh.write(f"{pattern}\t{text}\t{ident}\t{length}\t{label}\n")

    print(f"wrote {len(rows)} pairs to {path}")
    print()
    print("isD distribution (smax from harness):")
    print(summarize(rows, args.smax))

    if args.verify_determinism:
        rows2 = build(args.cases, args.seed)
        if rows2 != rows:
            print("\nDETERMINISM: FAILED -- regeneration differs")
            return 1
        print("\nDETERMINISM: ok (regeneration byte-identical)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
