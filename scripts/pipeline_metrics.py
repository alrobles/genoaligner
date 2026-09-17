#!/usr/bin/env python3
"""Pipeline production metrics: walltime per stage, before vs now.

Collects timing evidence for the phylogeny production pipeline:
  stage A  per-locus alignment   (MACSE jobs vs genomsa logs)
  stage B  tree inference        (mini_iqt / chains / caster / ft)
  stage C  topology evaluation   (au_test)

Sources:
  - genomsa per-locus logs:  "align total: X s (gpu)" + stage breakdown
  - MACSE: no per-locus timing in .macse.log; uses sacct Elapsed of the
    align_genes_* / align_mt_* jobs (they repeatedly hit the 6h wall)
  - CASTER run.meta.tsv:     elapsed_seconds, threads, slurm_max_rss
  - Slurm sacct:             job elapsed by name

Usage: pipeline_metrics.py [--pai DIR] [--out FILE] [--since YYYY-MM-DD]
Writes a TSV plus a markdown summary to stdout.
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("--pai", default="/beegfs/a474r867/phylogenyAI")
p.add_argument("--out", default=None, help="tsv output path")
p.add_argument("--since", default="2026-08-01")
a = p.parse_args()

PAI = a.pai
rows = []  # (stage, variant, item, metric, value, unit, source)


def add(stage, variant, item, metric, value, unit, source):
    rows.append((stage, variant, item, metric, value, unit, source))


# ---- stage A: genomsa per-locus alignment logs ----
for variant, d in (("base", "genomsa_codon"), ("lf", "genomsa_codon_lf"),
                   ("lf_gc1", "genomsa_codon_lf_gc1")):
    for lf in sorted(glob.glob(f"{PAI}/data/{d}/*.log")):
        gene = os.path.basename(lf)[:-4]
        txt = open(lf, errors="replace").read()
        m = re.search(r"align total: ([\d.]+)s", txt)
        if m:
            add("A_align", variant, gene, "walltime_s", m.group(1), "s", lf)
        st = re.search(r"dist=([\d.]+)s tree=([\d.]+)s.*align=([\d.]+)s", txt)
        if st:
            add("A_align", variant, gene, "dist_s", st.group(1), "s", lf)
            add("A_align", variant, gene, "tree_s", st.group(2), "s", lf)
            add("A_align", variant, gene, "align_s", st.group(3), "s", lf)

# ---- stage A: MACSE jobs via sacct ----
def sacct(names, since):
    try:
        out = subprocess.run(
            ["sacct", "-X", "--format=JobID,JobName,Elapsed,State",
             "-S", since, "--parsable2", "-n"],
            capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    recs = []
    for line in out.splitlines():
        f = line.split("|")
        if len(f) >= 4 and any(n in f[1] for n in names):
            recs.append({"job": f[0], "name": f[1],
                         "elapsed": f[2], "state": f[3]})
    return recs


def elapsed_s(el):
    # sacct format [DD-]HH:MM:SS; empty/malformed -> 0
    days, _, t = (el or "").partition("-")
    parts = t.split(":")
    if len(parts) != 3:
        return 0
    h, m, s = (int(x or 0) for x in parts)
    return (int(days or 0) * 86400) + h * 3600 + m * 60 + s


for r in sacct(("align_genes", "align_mt", "macse"), a.since):
    add("A_align", "macse", r["job"], "job_walltime_s",
        elapsed_s(r["elapsed"]), "s", f'sacct:{r["name"]}:{r["state"]}')

# ---- stage B: tree inference jobs ----
for r in sacct(("mini_iqt", "iqt_chain", "au_test", "ft_sm", "gsm_iqtree",
                "macse_iqtree", "codon_sm_iqtree", "sm_iqtree",
                "caster_full", "caster_mini"), a.since):
    add("B_tree", "slurm", r["job"], "job_walltime_s",
        elapsed_s(r["elapsed"]), "s", f'sacct:{r["name"]}:{r["state"]}')

# ---- stage B: caster meta ----
for meta in glob.glob(f"{PAI}/results/caster_backbone/**/run.meta.tsv",
                      recursive=True):
    kv = dict(r for r in csv.reader(open(meta), delimiter="\t") if len(r) == 2)
    if "elapsed_seconds" not in kv:
        continue
    scope = "mini" if "mini_v2" in meta else "full"
    parts = meta.split("/")
    variant = parts[-2]
    add("B_tree", f"caster_{scope}", f"{variant}:{parts[-3]}",
        "walltime_s", kv["elapsed_seconds"], "s", meta)
    if kv.get("slurm_max_rss"):
        add("B_tree", f"caster_{scope}", f"{variant}:{parts[-3]}",
            "max_rss_kb", kv["slurm_max_rss"].rstrip("K"), "KB", meta)

# ---- emit ----
w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
if a.out:
    out_fh = open(a.out, "w", newline="")
    w = csv.writer(out_fh, delimiter="\t", lineterminator="\n")
w.writerow(["stage", "variant", "item", "metric", "value", "unit", "source"])
for r in rows:
    w.writerow(r)

# ---- summary to stderr ----
def summ(pred, label):
    vals = [float(r[4]) for r in rows if pred(r)]
    if vals:
        print(f"{label}: n={len(vals)} total={sum(vals):.0f}s "
              f"median={sorted(vals)[len(vals)//2]:.1f}s "
              f"max={max(vals):.0f}s", file=sys.stderr)

print("\n== SUMMARY ==", file=sys.stderr)
summ(lambda r: r[0] == "A_align" and r[1] == "base" and r[3] == "walltime_s",
     "genomsa base per-locus align")
summ(lambda r: r[0] == "A_align" and r[1] == "lf" and r[3] == "walltime_s",
     "genomsa lf per-locus align")
summ(lambda r: r[0] == "A_align" and r[1] == "macse",
     "macse align jobs (sacct, incl. timed-out)")
summ(lambda r: r[0] == "B_tree" and r[1] == "caster_mini",
     "caster mini runs")
summ(lambda r: r[0] == "B_tree" and r[1] == "caster_full",
     "caster full runs")
