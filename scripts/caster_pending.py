#!/usr/bin/env python3
"""Print incomplete CASTER array indices for restart-safe submission.

A finished cell is skipped only when its manifest verifies: a valid
treefile plus a complete run whose recorded input hash still matches
the input file on disk. Legacy results without hashes are pending --
they are archived (not deleted) by caster_run.sh on the rerun.
"""
import argparse
import importlib.util
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "caster_report", os.path.join(SCRIPT_DIR, "caster_report.py"))
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


FULL_VARIANTS = ("genomsa", "genomsa_lf", "codon")
MINI_VARIANTS = ("macse", "base", "lf")
MINI_REPS = 5


def cell_paths(root, scope, index):
    """Return (input, outdir) mirroring the array sbatch mapping."""
    if scope == "full":
        variant = FULL_VARIANTS[index]
        return (
            os.path.join(
                root, "data", f"supermatrix_{variant}", "supermatrix.fasta"),
            os.path.join(
                root, "results", "caster_backbone", "full", variant),
        )
    if scope == "mini":
        rep = index // len(MINI_VARIANTS)
        variant = MINI_VARIANTS[index % len(MINI_VARIANTS)]
        return (
            os.path.join(
                root, "results", "mini_backbone_v2",
                f"rep{rep}", variant, "supermatrix.fasta"),
            os.path.join(
                root, "results", "caster_backbone", "mini_v2",
                f"rep{rep}", variant),
        )
    raise ValueError(f"unknown scope: {scope}")


def cell_count(scope):
    if scope == "full":
        return len(FULL_VARIANTS)
    if scope == "mini":
        return MINI_REPS * len(MINI_VARIANTS)
    raise ValueError(f"unknown scope: {scope}")


def cell_states(root, scope):
    states = []
    for index in range(cell_count(scope)):
        input_path, outdir = cell_paths(root, scope, index)
        states.append((index, REPORT.result_state(outdir, input_path)))
    return states


def pending_indices(root, scope):
    return [
        index
        for index, state in cell_states(root, scope)
        if state != "verified_complete"
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/beegfs/a474r867/phylogenyAI")
    parser.add_argument("--scope", choices=("full", "mini"), required=True)
    parser.add_argument(
        "--explain", action="store_true",
        help="print 'index<TAB>state' per cell instead of the index list")
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    states = cell_states(root, args.scope)
    if args.explain:
        for index, state in states:
            print(f"{index}\t{state}")
    else:
        print(",".join(
            str(index) for index, state in states
            if state != "verified_complete"))


if __name__ == "__main__":
    main()
