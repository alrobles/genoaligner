#!/usr/bin/env python3
"""codon_smoke_report — aggregate per-instance results.tsv files from the
smoke grid into per-cell x per-tool summary statistics with bootstrap CIs.

Reads every $OUT/<cell>/r<k>/results.tsv, groups by (cell, tool), and
reports mean + 95% bootstrap CI over replicate instances (percentile
method, 10k resamples, seed 0). Failures/timeouts are counted and shown.

Usage: codon_smoke_report.py $OUTDIR > report.md
"""
import glob
import json
import math
import os
import random
import sys


def boot_ci(xs, seed=0, n_boot=10000):
    """Percentile bootstrap 95% CI of the mean over instances."""
    xs = [x for x in xs if x == x]           # drop nan
    if not xs:
        return float("nan"), float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, m, m
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(rng.choice(xs) for _ in xs) / len(xs))
    means.sort()
    return m, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main():
    out = sys.argv[1]
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "*", "r*", "results.tsv"))):
        for line in open(p):
            f = line.rstrip("\n").split("\t")
            if len(f) < 13:
                continue
            rows.append({
                "cell": f[0], "rep": int(f[1]), "n": int(f[2]),
                "L": int(f[3]), "gc": int(f[4]), "tool": f[5],
                "sps": float(f[6]), "tc": float(f[7]),
                "w_got": float(f[8]), "w_true": float(f[9]),
                "purity": float(f[10]), "wall": float(f[11]),
                "status": f[12], "dir": os.path.dirname(p)})

    cells = sorted({r["cell"] for r in rows})
    tools = ["genomsa", "macse", "prank", "threestep"]

    def fmt(v):
        return "—" if v != v else f"{v:.3f}"

    print("# codon_smoke report\n")
    print(f"instances found: {len(rows)}\n")

    # expected reps from grid
    grid_p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "codon_smoke_grid.json")
    exp = {}
    if os.path.exists(grid_p):
        g = json.load(open(grid_p))
        exp = {c["id"]: g["reps"] for c in g["cells"]}

    print("## SPS (column sum-of-pairs vs true MSA)\n")
    print("| cell | tool | n_inst | SPS | 95% CI | TC | wall s | fails |")
    print("|---|---|---|---|---|---|---|---|")
    for cell in cells:
        for tool in tools:
            sub = [r for r in rows if r["cell"] == cell
                   and r["tool"] == tool]
            if not sub:
                continue
            ok = [r for r in sub if r["status"] == "ok"]
            fails = len(sub) - len(ok)
            sps, lo, hi = boot_ci([r["sps"] for r in ok])
            tc, _, _ = boot_ci([r["tc"] for r in ok])
            wall, _, _ = boot_ci([r["wall"] for r in ok])
            print(f"| {cell} | {tool} | {len(ok)}/{len(sub)} | "
                  f"{fmt(sps)} | [{fmt(lo)},{fmt(hi)}] | {fmt(tc)} | "
                  f"{wall:.0f} | {fails} |")
        print("|---|---|---|---|---|---|---|---|")

    print("\n## Walltime (s, mean over ok instances)\n")
    print("| cell | " + " | ".join(tools) + " |")
    print("|---|" + "---|" * len(tools))
    for cell in cells:
        vals = []
        for tool in tools:
            ok = [r["wall"] for r in rows if r["cell"] == cell
                  and r["tool"] == tool and r["status"] == "ok"]
            vals.append(f"{sum(ok)/len(ok):.0f}" if ok else "—")
        print(f"| {cell} | " + " | ".join(vals) + " |")

    print("\n## Failure inventory\n")
    for r in rows:
        if r["status"] != "ok":
            print(f"- {r['cell']}/r{r['rep']} {r['tool']}: {r['status']} "
                  f"({r['dir']})")


if __name__ == "__main__":
    main()
