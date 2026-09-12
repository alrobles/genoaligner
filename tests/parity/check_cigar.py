#!/usr/bin/env python3
"""Fase 4: compare our CIGAR against edlib's, on SCORE (not on operations).

WHY THIS EXISTS
---------------
The self-consistency checks in traceback_cpu.cpp (re-score + well-formedness)
can both pass while the CIGAR is still wrong, and that actually happened during
development. edlib is an independent implementation that also produces a CIGAR
(task="path"), so its alignment is a genuinely external reference.

WHAT IS COMPARED, AND WHAT IS NOT
---------------------------------
Scores are compared; operation STRINGS are not. Multiple distinct alignments are
optimal, and edlib will legitimately pick a different one. Demanding identical
strings would produce spurious failures and invite a wrong "fix" -- exactly the
trap the plan warns about.

So the check is:
    score(cigar_from_edlib) == score(cigar_from_us)
where score is computed by the same one-character-one-cost rule:
    M consumes 1+1, X consumes 1+1 and costs 1, I consumes 1 text and costs 1,
    D consumes 1 pattern and costs 1.

A mismatch here would mean our score and edlib's disagree, which is the real
signal.

USAGE
    python3 check_cigar.py <emit.tsv> [--limit N]
where emit.tsv has: index<TAB>pattern<TAB>text<TAB>gpu<TAB>cpu<TAB>... and
column 4 (gpu) carries OUR CIGAR string when the harness runs in --traceback
mode.

NOTE: importable so traceback_cpu can shell out to it, but also usable
stand-alone for investigation.
"""
import argparse
import sys

try:
    import edlib
except ImportError:
    print("ERROR: edlib missing. pip install -r tests/parity/requirements-oracle.txt",
          file=sys.stderr)
    sys.exit(2)


def score_cigar(cigar: str) -> int:
    """Cost of a CIGAR over {M,X,I,D}: X, I and D each cost 1."""
    return sum(1 for op in cigar if op in "XID")


def edlib_cigar_score(pattern: str, text: str):
    """Score edlib's optimal CIGAR. Returns (score, cigar)."""
    r = edlib.align(pattern, text, mode="NW", task="path")
    return r["editDistance"], r["cigar"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsv", help="emit file whose column 4 holds our CIGAR")
    ap.add_argument("--limit", type=int, default=0, help="check only first N cases")
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
            if len(p) < 4:
                continue
            rows.append(p)

    checked = agree = disagree = skipped = 0
    examples = []
    for p in rows:
        pattern, text = p[1], p[2]
        our_cigar = p[3]
        if our_cigar in ("", "-") and not (pattern == "" and text == ""):
            skipped += 1
            continue
        if args.limit and checked >= args.limit:
            break

        ours = score_cigar(our_cigar)
        theirs, their_cigar = edlib_cigar_score(pattern, text)

        checked += 1
        if ours == theirs:
            agree += 1
        else:
            disagree += 1
            if len(examples) < 10:
                examples.append((pattern[:24], text[:24], our_cigar,
                                 ours, their_cigar, theirs))

    print("Fase 4: CIGAR score vs edlib (independent implementation)")
    print(f"  rows read       : {len(rows)}")
    print(f"  checked         : {checked}")
    print(f"  skipped         : {skipped}  (no CIGAR emitted)")
    print(f"  score agrees    : {agree}")
    print(f"  DISAGREES       : {disagree}")
    if examples:
        print()
        print("  disagreements (ours vs edlib):")
        for pat, txt, oc, os_, tc, ts in examples:
            print(f"    {pat!r} vs {txt!r}")
            print(f"      ours  : {oc} = {os_}")
            print(f"      edlib : {tc} = {ts}")
    print()
    if checked == 0:
        print("  RESULT: INCONCLUSIVE -- nothing was checked")
        return 2
    if disagree:
        print("  RESULT: FAIL -- our alignment score disagrees with edlib")
        return 1
    print("  RESULT: PASS -- alignment scores identical to edlib")
    return 0


if __name__ == "__main__":
    sys.exit(main())
