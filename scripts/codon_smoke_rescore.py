#!/usr/bin/env python3
"""codon_smoke_rescore — re-score smoke-grid instances whose results.tsv
row carries a scoring failure, without re-running the aligner.

For every $OUT/<cell>/r<k>/results.tsv row with status in {score-fail},
look for that tool's output artefact in the workdir and re-run
codon_sim_v2.py in score mode using the instance parameters recorded in
manifest.json (seed included, so the regenerated truth is identical).
On success the row's sps/tc/widths/purity are replaced and status set to
ok; the original walltime is kept (the alignment is not re-run).

Also relabels `rc=0` statuses to `no-output` (the post-257ee6a name for
"tool exited 0 but wrote no alignment" — e.g. PRANK rejecting input that
is not a multiple of 3).

Usage: codon_smoke_rescore.py $OUTDIR [--write]
Without --write, prints a dry-run diff and changes nothing.
"""
import glob
import json
import os
import re
import subprocess
import sys

SIM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "codon_sim_v2.py")

ARTEFACT = {"genomsa": "genomsa.fasta", "macse": "macse_nt.fasta",
            "prank": "prank.best.fas", "threestep": "ts_got.fasta"}

PAT = re.compile(r"SIM-SPS ([\d.]+) SIM-TC ([\d.]+) width_true=(\d+) "
                 r"width_got=(\d+) trip_purity=([\d.]+)")


def sim_args(man):
    return [str(man["n_leaves"]), str(man["seed"]), str(man["sub_rate"]),
            str(man["indel_rate"]), str(man["L"]),
            "--omega", str(man["omega"]),
            "--fs-frac", str(man["fs_frac"]),
            "--stop-frac", str(man["stop_frac"]),
            "--frag5", str(man["frag5"]),
            "--frag3", str(man["frag3"]),
            "--err", str(man["err"]),
            "--lr-frac", str(man["lr_frac"]),
            "--lr-mult", str(man.get("lr_mult", 5.0)),
            "--gc", str(man["gc"])]


def main():
    out = sys.argv[1]
    write = "--write" in sys.argv[2:]
    n_rescored = n_still_failing = n_relabel = 0

    for tsv in sorted(glob.glob(os.path.join(out, "*", "r*",
                                             "results.tsv"))):
        wd = os.path.dirname(tsv)
        mp = os.path.join(wd, "manifest.json")
        if not os.path.exists(mp):
            continue
        man = json.load(open(mp))
        lines = open(tsv).read().splitlines()
        changed = False
        for i, line in enumerate(lines):
            f = line.split("\t")
            if len(f) < 13:
                continue
            tool, status = f[5], f[12]
            if status == "rc=0":
                f[12] = "no-output"
                lines[i] = "\t".join(f)
                changed = True
                n_relabel += 1
                print(f"relabel  {wd} {tool}: rc=0 -> no-output")
                continue
            if status != "score-fail":
                continue
            art = os.path.join(wd, ARTEFACT.get(tool, ""))
            if not os.path.exists(art):
                print(f"no-art   {wd} {tool}: artefact missing")
                continue
            p = subprocess.run([sys.executable, SIM, art] + sim_args(man),
                               capture_output=True, text=True, cwd=wd)
            m = PAT.search(p.stdout)
            if not m:
                n_still_failing += 1
                print(f"still-fail {wd} {tool}: "
                      f"{p.stderr.strip().splitlines()[-1] if p.stderr else 'no score output'}")
                continue
            f[6], f[7], f[8], f[9], f[10], f[12] = (
                m.group(1), m.group(2), m.group(4), m.group(3),
                m.group(5), "ok")
            lines[i] = "\t".join(f)
            changed = True
            n_rescored += 1
            print(f"rescored {wd} {tool}: sps={m.group(1)} "
                  f"tc={m.group(2)}")
        if changed and write:
            with open(tsv, "w") as fh:
                fh.write("\n".join(lines) + "\n")

    print(f"\n{'' if write else '[dry-run] '}rescored={n_rescored} "
          f"still-failing={n_still_failing} relabeled={n_relabel}")


if __name__ == "__main__":
    main()
