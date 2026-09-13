#!/usr/bin/env python3
# Aggregate benchmark report: scores every <method>_out/<set>/<family>
# alignment against bench1.0 refs, joins per-family runtimes, and emits
# per-set mean/median SP/TC, BAliBASE per-category rows, SABmark
# twilight/superfamily split, and a method x (SP, time) tradeoff table.
#
#   msa_bench_report.py BASE_DIR BENCH1.0_DIR RV_MAP.tsv
# BASE_DIR contains <method>_out/, *_times.tsv; RV_MAP maps BB->category.
import os
import re
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msa_bench_score import score, read_fasta

BASE, BENCH, RVMAP = sys.argv[1], sys.argv[2], sys.argv[3]
SETS = ["bali3", "sabrem", "ox", "prefab4"]

cats = {}
for line in open(RVMAP):
    p = line.rstrip("\n").split("\t")
    if len(p) == 2:
        cats[p[0]] = p[1]


def subset_of(name, dset):
    if dset == "bali3":
        return cats.get(name, "?")
    if dset in ("sabrem", "sabre"):
        return name.split("_")[0]          # sup / twi
    return "all"


def load_times():
    t = {}
    for fn in os.listdir(BASE):
        if fn.endswith("_times.tsv"):
            for line in open(os.path.join(BASE, fn)):
                p = line.rstrip("\n").split("\t")
                if len(p) >= 5:
                    t[(p[0], p[1], p[2])] = (float(p[3]), p[4])
    return t


def methods():
    out = set()
    for fn in os.listdir(BASE):
        m = re.match(r"(.+)_out$", fn)
        if m:
            out.add(m.group(1))
    return sorted(out)


times = load_times()
# {method: {dset: {subset: [(fam, sp, tc, secs)]}}}
data = {}
for m in methods():
    data[m] = {d: {} for d in SETS}
    for d in SETS:
        odir = os.path.join(BASE, f"{m}_out", d)
        rdir = os.path.join(BENCH, d, "ref")
        if not os.path.isdir(odir):
            continue
        for fam in sorted(os.listdir(rdir)):
            tp = os.path.join(odir, fam)
            if not os.path.exists(tp):
                continue
            try:
                sp, tc, _, _, _ = score(read_fasta(tp),
                                        read_fasta(os.path.join(rdir, fam)))
            except Exception:
                continue
            tkey = "genomsa_cpu" if m == "genomsa" else m
            secs = times.get((tkey, d, fam), (float("nan"), "?"))[0]
            sub = subset_of(fam, d)
            data[m][d].setdefault(sub, []).append((fam, sp, tc, secs))

hdr = f"{'method':14s} {'set':7s} {'subset':6s} {'n':>5s} " \
      f"{'SPmean':>7s} {'SPmed':>7s} {'TCmean':>7s} {'TCmed':>7s} " \
      f"{'Ttot':>9s}"
print(hdr)
print("-" * len(hdr))
for m in methods():
    for d in SETS:
        for sub in sorted(data[m][d]):
            rows = data[m][d][sub]
            if not rows:
                continue
            sp = [r[1] for r in rows]
            tc = [r[2] for r in rows]
            tt = sum(r[3] for r in rows if r[3] == r[3])
            print(f"{m:14s} {d:7s} {sub:6s} {len(rows):5d} "
                  f"{statistics.fmean(sp):7.4f} {statistics.median(sp):7.4f} "
                  f"{statistics.fmean(tc):7.4f} {statistics.median(tc):7.4f} "
                  f"{tt:9.1f}")
    print()

print("\n== tradeoff (set x method: mean SP / total wall-seconds) ==")
print(f"{'set':8s} {'method':14s} {'SP':>7s} {'TC':>7s} {'Ttot_s':>10s}")
for d in SETS:
    for m in methods():
        rows = [r for sub in data[m][d].values() for r in sub]
        if not rows:
            continue
        sp = statistics.fmean(r[1] for r in rows)
        tc = statistics.fmean(r[2] for r in rows)
        tt = sum(r[3] for r in rows if r[3] == r[3])
        print(f"{d:8s} {m:14s} {sp:7.4f} {tc:7.4f} {tt:10.1f}")
