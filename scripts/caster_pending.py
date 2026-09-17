#!/usr/bin/env python3
"""Print incomplete CASTER array indices for restart-safe submission."""
import argparse
import os


FULL_VARIANTS = ("genomsa", "genomsa_lf", "codon")
MINI_VARIANTS = ("macse", "base", "lf")


def tree_complete(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    with open(path) as handle:
        return ";" in handle.read()


def pending_indices(root, scope):
    result = []
    if scope == "full":
        for index, variant in enumerate(FULL_VARIANTS):
            tree = os.path.join(
                root, "results", "caster_backbone", "full", variant,
                "caster.treefile")
            if not tree_complete(tree):
                result.append(index)
        return result
    if scope == "mini":
        for index in range(15):
            rep = index // len(MINI_VARIANTS)
            variant = MINI_VARIANTS[index % len(MINI_VARIANTS)]
            tree = os.path.join(
                root, "results", "caster_backbone", "mini_v2",
                f"rep{rep}", variant, "caster.treefile")
            if not tree_complete(tree):
                result.append(index)
        return result
    raise ValueError(f"unknown scope: {scope}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/beegfs/a474r867/phylogenyAI")
    parser.add_argument("--scope", choices=("full", "mini"), required=True)
    args = parser.parse_args()
    print(",".join(str(index) for index in pending_indices(
        os.path.abspath(args.root), args.scope)))


if __name__ == "__main__":
    main()
