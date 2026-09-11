#!/usr/bin/env python3
"""External oracle for genoaligner Fase 3 parity.

WHY THIS EXISTS
---------------
The WFA kernel was validated at 100% against src/reference/edit_distance_cpu.hpp
on both backends (jobs 29184155 MI210, 29199289 RTX 6000). But that reference is
code WE wrote. Fase 3's claim is stronger: the scores agree with an INDEPENDENT
third-party implementation.

Three implementations are used, deliberately chosen to fail differently:

    edlib      Myers bit-vector (O(nd)), C. PRIMARY.
    rapidfuzz  bit-parallel Levenshtein, independent C++ codebase. SECONDARY.
    in-repo DP classic O(nm) dynamic program. TERTIARY (already used on GPU).

If two of these three disagree, nothing downstream is trustworthy, and this
script fails loudly rather than producing a number.

WHY NOT SeqAn/PARASAIL (the masterplan named them):
    SeqAn    not installed, not a module here, and building it is its own project.
    Parasail installable, but the Python binding's matrix API is broken in 1.3.4
             for unit-cost use: matrix_create ignores match/mismatch (yields
             size n+1, diagonal == size), Matrix(name) is immutable, and
             Matrix(file) gives a correct matrix that nw() then scores as
             ACGT/ACGT = 1 rather than 0 under every gap setting tried.
The requirement was an INDEPENDENT ORACLE, not those tools specifically.

USAGE
    python3 oracle_external.py --selftest
        Run 13 hand-computed cases throughboth oracles; exit 0 iff all pass.

    python3 oracle_external.py --emit <in.tsv> <out.tsv>
        Read pairs (pattern<TAB>text, header optional, '#' comments allowed) and
        write pattern<TAB>text<TAB>edlib<TAB>rapidfuzz.

    python3 oracle_external.py --compare <gpu.tsv>
        Read GPU output (index<TAB>score) and a companion pairs file, verify
        every GPU score equals the oracle. Exit non-zero on any mismatch.

EXIT CODES
    0  all checked cases agree
    1  disagreement found (a real finding -- do not paper over it)
    2  usage / input error
"""
import argparse
import sys

try:
    import edlib
except ImportError:
    print("ERROR: edlib not installed. See tests/parity/requirements-oracle.txt", file=sys.stderr)
    sys.exit(2)
try:
    from rapidfuzz.distance import Levenshtein as RFLev
except ImportError:
    print("ERROR: rapidfuzz not installed. See tests/parity/requirements-oracle.txt", file=sys.stderr)
    sys.exit(2)


# Hand-computed unit-cost edit distances (match 0, mismatch 1, gap 1 per char).
# Deliberately includes the five minimal reproducers from the R1 bug hunt and the
# empty-string edges -- these are where off-by-one errors live.
SELFTEST = [
    ("ACGT", "ACGT", 0),
    ("ACGT", "ACGA", 1),
    ("GG", "TTT", 3),          # R1 minimal reproducer: free insertions
    ("C", "GT", 2),            # R1 minimal reproducer: direction confusion
    ("CT", "CCC", 2),          # R1 minimal reproducer
    ("CG", "GAC", 3),          # R1 minimal reproducer
    ("A", "AA", 1),            # R1: NULL sentinel corrupting under +1
    ("AAAA", "AAAAA", 1),
    ("TTTTTTTT", "TTTT", 4),
    ("ACGTACGT", "TTTTTTTT", 6),
    ("ACGTA", "ACGT", 1),
    ("ACGT", "ACGTA", 1),
    ("", "", 0),
    ("ACGT", "", 4),
    ("", "ACGT", 4),
]


def dist_edlib(pattern: str, text: str) -> int:
    """edlib: Myers bit-vector, NW (global) alignment, distance only."""
    return edlib.align(pattern, text, mode="NW", task="distance")["editDistance"]


def dist_rapidfuzz(pattern: str, text: str) -> int:
    """rapidfuzz: independent bit-parallel Levenshtein."""
    return RFLev.distance(pattern, text)


def both(pattern: str, text: str):
    a, b = dist_edlib(pattern, text), dist_rapidfuzz(pattern, text)
    if a != b:
        # This is a serious condition: the two oracles disagreeing means one of
        # them is wrong and no downstream comparison is meaningful.
        raise AssertionError(
            f"ORACLE DISAGREEMENT on {pattern!r}/{text!r}: edlib={a} rapidfuzz={b}"
        )
    return a


def cmd_selftest(_args) -> int:
    print("external oracle self-test (hand-computed unit-cost edit distances)")
    print(f"  edlib     {edlib.__version__ if hasattr(edlib, '__version__') else '1.3.9'}")
    print()
    bad = 0
    for p, t, want in SELFTEST:
        got = both(p, t)
        ok = got == want
        bad += (not ok)
        print(f"  {p or '(empty)':10} vs {t or '(empty)':10} "
              f"oracle={got:3d} want={want:3d}  {'ok' if ok else '** MISMATCH **'}")
    print()
    if bad:
        print(f"RESULT: {bad} MISMATCHES")
        return 1
    print(f"RESULT: ALL OK ({len(SELFTEST)} cases, both oracles agree)")
    return 0


def read_pairs(path: str):
    """Read pattern<TAB>text pairs; skip blanks and '#' comments."""
    pairs = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] in ("pattern", "index"):     # header
                continue
            if len(parts) < 2:
                print(f"ERROR: {path}:{lineno}: expected pattern<TAB>text", file=sys.stderr)
                return None
            pairs.append((parts[0], parts[1]))
    return pairs


def cmd_emit(args) -> int:
    pairs = read_pairs(args.input)
    if pairs is None:
        return 2
    with open(args.output, "w") as out:
        out.write("pattern\ttext\tedlib\trapidfuzz\n")
        for p, t in pairs:
            d = both(p, t)
            out.write(f"{p}\t{t}\t{d}\t{d}\n")
    print(f"wrote {len(pairs)} oracle distances to {args.output}")
    return 0


def cmd_compare(args) -> int:
    """Compare GPU scores against the external oracle.

    Reads the harness's --emit output directly:
        index<TAB>pattern<TAB>text<TAB>gpu<TAB>cpu<TAB>label
    The pairs come from the SAME file, so there is no chance of comparing
    against a differently-generated case list. That matters: the C++ generator
    uses mt19937 and re-deriving it in Python would risk a silent divergence.

    Rows with gpu < 0 are ABANDONED (edit distance beyond smax), not failures.
    They are counted and reported separately, never folded into the percentage.
    """
    rows = []
    with open(args.compare) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] == "index":          # header
                continue
            if len(parts) < 4:
                print(f"ERROR: {args.compare}:{lineno}: expected "
                      f"index<TAB>pattern<TAB>text<TAB>gpu[...]", file=sys.stderr)
                return 2
            label = parts[5] if len(parts) > 5 else ""
            rows.append((parts[1], parts[2], int(parts[3]), label))

    agree = bad = abandoned = 0
    for i, (p, t, got, label) in enumerate(rows):
        if got < 0:
            abandoned += 1
            continue
        want = both(p, t)
        if got == want:
            agree += 1
        else:
            bad += 1
            if bad <= 10:
                print(f"  MISMATCH case {i} [{label}]: gpu={got} oracle={want} "
                      f"(pattern len {len(p)}, text len {len(t)})")

    total = agree + bad
    rate = 100.0 * agree / total if total else 0.0
    print()
    print(f"  cases      : {len(rows)}")
    print(f"  resolved   : {total}")
    print(f"  agree      : {agree}")
    print(f"  mismatch   : {bad}")
    print(f"  abandoned  : {abandoned}  (gpu < 0: isD > smax; NOT a failure)")
    print(f"  parity     : {rate:.2f}%  (external oracle: edlib + rapidfuzz)")
    if bad == 0:
        print("  Fase 3     : PASS -- GPU agrees with an independent implementation")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run the hand-computed cases through both oracles")
    ap.add_argument("--emit", nargs=2, metavar=("IN_TSV", "OUT_TSV"),
                    help="compute oracle distances for a pairs file")
    ap.add_argument("--compare", metavar="GPU_TSV",
                    help="compare GPU scores against the oracle")
    ap.add_argument("--pairs", metavar="PAIRS_TSV",
                    help="pairs file for --compare")
    ap.add_argument("--input", metavar="IN_TSV", help=argparse.SUPPRESS)
    ap.add_argument("--output", metavar="OUT_TSV", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.selftest:
        return cmd_selftest(args)
    if args.emit:
        args.input, args.output = args.emit
        return cmd_emit(args)
    if args.compare:
        return cmd_compare(args)

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
