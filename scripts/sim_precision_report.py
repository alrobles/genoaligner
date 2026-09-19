#!/usr/bin/env python3
"""sim_precision_report — aggregate tree-precision results.

Reads every $OUT/<cell>/r<k>/results.tsv and reports per (cell, tool):
n_ok, mean SPS, mean RF_norm vs the true simulated tree, and the paired
win-rate of each genomsa variant vs macse on the same instances. The
`truesim` rows give the FastTree noise floor per cell.

Usage: sim_precision_report.py $OUTDIR
"""
import glob
import os
import statistics as st
import sys


def rows(path):
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 15 or f[0] == "cell":
            continue
        yield {"cell": f[0], "rep": f[1], "tool": f[5],
               "sps": float(f[6]), "tc": float(f[7]),
               "status": f[12], "rf": float(f[13]),
               "rfn": float(f[14])}


def main():
    out = sys.argv[1]
    data = {}
    for tsv in sorted(glob.glob(os.path.join(out, "*", "r*",
                                             "results.tsv"))):
        for r in rows(tsv):
            data.setdefault((r["cell"], r["rep"]), []).append(r)

    cells = {}
    for (cell, rep), rs in data.items():
        cells.setdefault(cell, {})[rep] = {r["tool"]: r for r in rs}

    hdr = (f"{'cell':<12} {'tool':<10} {'n':>3} {'sps':>7} "
           f"{'rf_mean':>8} {'rf_min':>7} {'rf_max':>7} "
           f"{'floor':>7} {'d_floor':>8}")
    print(hdr)
    for cell in sorted(cells):
        reps = cells[cell]
        tools = sorted({t for rs in reps.values() for t in rs})
        for tool in tools:
            rfs = [rs[tool]["rfn"] for rs in reps.values()
                   if tool in rs and rs[tool]["status"] == "ok"
                   and rs[tool]["rfn"] == rs[tool]["rfn"]]
            sps = [rs[tool]["sps"] for rs in reps.values()
                   if tool in rs and rs[tool]["status"] == "ok"]
            floor = [rs["truesim"]["rfn"] for rs in reps.values()
                     if "truesim" in rs]
            if not rfs:
                print(f"{cell:<12} {tool:<10} {0:>3}")
                continue
            fl = st.mean(floor) if floor else float("nan")
            print(f"{cell:<12} {tool:<10} {len(rfs):>3} "
                  f"{st.mean(sps):>7.4f} {st.mean(rfs):>8.4f} "
                  f"{min(rfs):>7.4f} {max(rfs):>7.4f} "
                  f"{fl:>7.4f} {st.mean(rfs) - fl:>8.4f}")
        # paired win-rates vs macse and base-vs-lf
        for a, b in (("genomsa_lf", "macse"), ("genomsa", "macse"),
                     ("genomsa_lf", "genomsa")):
            wins = ties = losses = 0
            for rs in reps.values():
                if a in rs and b in rs and rs[a]["status"] == "ok" \
                        and rs[b]["status"] == "ok":
                    da, db = rs[a]["rfn"], rs[b]["rfn"]
                    if da != da or db != db:
                        continue
                    wins += da < db
                    ties += da == db
                    losses += da > db
            if wins + ties + losses:
                print(f"{'':>13}{a} vs {b}: "
                      f"win={wins} tie={ties} loss={losses}")
        print()


if __name__ == "__main__":
    main()
