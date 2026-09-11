#!/usr/bin/env python3
"""Cross-check all three oracles against each other on the same pairs.

Fase 3's claim is that the GPU agrees with an INDEPENDENT implementation. That
claim is only as strong as the oracles' agreement with each other: if edlib,
rapidfuzz and SeqAn3 disagree on a pair, then one of them is wrong and no
downstream comparison means anything.

This script reads a SeqAn3-augmented emit file
    index <TAB> pattern <TAB> text <TAB> gpu <TAB> cpu <TAB> seqan3 <TAB> label
and verifies edlib == rapidfuzz == seqan3 on every row.

Exit 0 iff all three agree everywhere.
"""
import argparse
import sys

try:
    import edlib
    from rapidfuzz.distance import Levenshtein as RFLev
except ImportError as e:
    print(f"ERROR: missing oracle dependency: {e}", file=sys.stderr)
    print("  pip install -r tests/parity/requirements-oracle.txt", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsv", help="SeqAn3-augmented emit file (6+ columns)")
    ap.add_argument("--limit", type=int, default=0, help="only first N scored rows")
    args = ap.parse_args()

    rows = []
    with open(args.tsv) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            p = line.split("\t")
            if p[0] == "index":
                continue
            if len(p) < 6:
                continue
            rows.append(p)

    agree = disagree = skipped = 0
    examples = []
    for p in rows:
        pattern, text = p[1], p[2]
        gpu = int(p[3])
        seqan = int(p[5])

        # Unresolved on GPU (isD > smax) and unparsable rows are excluded.
        if gpu < 0 or seqan < 0:
            skipped += 1
            continue
        if args.limit and agree + disagree >= args.limit:
            break

        e = edlib.align(pattern, text, mode="NW", task="distance")["editDistance"]
        r = RFLev.distance(pattern, text)

        if e == r == seqan:
            agree += 1
        else:
            disagree += 1
            if len(examples) < 10:
                examples.append((pattern[:20], text[:20], e, r, seqan))

    print("three-oracle cross-check")
    print(f"  rows read        : {len(rows)}")
    print(f"  scored           : {agree + disagree}")
    print(f"  skipped          : {skipped}  (gpu<0 or seqan<0)")
    print(f"  all three agree  : {agree}")
    print(f"  DISAGREE         : {disagree}")
    if examples:
        print()
        print("  disagreements (edlib / rapidfuzz / seqan3):")
        for p, t, e, r, s in examples:
            print(f"    {p!r} vs {t!r}: {e} / {r} / {s}")
    print()
    if disagree:
        print("  RESULT: FAIL -- oracles disagree, nothing downstream is trustworthy")
        return 1
    print("  RESULT: PASS -- edlib == rapidfuzz == seqan3 on every scored row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
