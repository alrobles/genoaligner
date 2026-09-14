#!/usr/bin/env python3
"""codon_bench_grid — run one benchmark instance: every enabled tool on the
same simulated input, scored against the tracked truth by codon_sim_v2.

One invocation = one (cell, replicate). Prints one TSV row:
    cell rep n L gc tool sps tc width_got width_true purity wall_s status

Tools (each optional, discovered on PATH):
    genomsa    genomsa --cpu|--gpu --codon --gc-def GC
    macse      macse -prog alignSequences -seq IN -out_NT OUT
               (-max_refine_iter 0 unless --macse-refine)
    prank      prank -d=IN -o=OUT -codon [-F]
    threestep  MAFFT on translated AA -> back-translate (TranslatorX-analog;
               sequences with internal stops fail by construction)

Each instance dir holds sim_in.fasta, sim_true.fasta, manifest.json,
events.tsv, per-tool outputs and a status line, so every number in the
TSV is traceable to raw artefacts.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

SIM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "codon_sim_v2.py")

STOPS1 = {"TAA", "TAG", "TGA"}
STOPS2 = STOPS1 - {"TGA"} | {"AGA", "AGG"}
TAB1 = {}
_AA1 = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
_AA2 = ("FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSS**VVVVAAAADDEEGGGG")
for _i in range(64):
    _c = "TCAG"[_i // 16] + "TCAG"[(_i // 4) % 4] + "TCAG"[_i % 4]
    TAB1[_c] = _AA1[_i]
TAB2 = dict(TAB1)
for _i in range(64):
    _c = "TCAG"[_i // 16] + "TCAG"[(_i // 4) % 4] + "TCAG"[_i % 4]
    TAB2[_c] = _AA2[_i]


def read_fasta(path):
    d, cur = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            cur = line[1:].split()[0]
            d[cur] = ""
        elif cur:
            d[cur] += line.upper()
    return d


def run(cmd, timeout, log, **kw):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, **kw)
        dt = time.time() - t0
        if log:
            log.write(f"$ {' '.join(str(c) for c in cmd)}\n"
                      f"# {dt:.1f}s rc={p.returncode}\n{p.stderr[-2000:]}\n")
        return dt, p.returncode
    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        if log:
            log.write(f"$ {' '.join(str(c) for c in cmd)}\n# TIMEOUT {dt:.0f}s\n")
        return dt, -9


def score(got_path, sim_args, n):
    cmd = [sys.executable, SIM, got_path] + [str(a) for a in sim_args]
    p = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"SIM-SPS (\S+) SIM-TC (\S+) width_true=(\d+) "
                  r"width_got=(\d+) trip_purity=(\S+)", p.stdout)
    if not m:
        return None
    return {"sps": m.group(1), "tc": m.group(2),
            "w_true": m.group(3), "w_got": m.group(4),
            "purity": m.group(5)}


def threestep(seqs, wd, gc, timeout, log):
    """MAFFT on translated amino acids -> deterministic back-translate.
    Sequences with an internal stop are left unaligned-in-codon: they are
    emitted gap-only columns (the documented failure mode of 3-step
    translation-align on ORF-broken input, cf. MACSE's motivating case)."""
    tab = TAB1 if gc == 1 else TAB2
    stops = STOPS1 if gc == 1 else STOPS2
    aa_in = os.path.join(wd, "ts_aa_in.fasta")
    aa_aln = os.path.join(wd, "ts_aa.fasta")
    bad = set()
    with open(aa_in, "w") as f:
        for name, s in seqs.items():
            aa = []
            for i in range(0, len(s) - 2, 3):
                c = s[i:i + 3]
                a = tab.get(c, "X")
                aa.append(a)
            if "*" in "".join(aa):
                bad.add(name)
            f.write(f">{name}\n{''.join(aa)}\n")
    # mafft writes the alignment to stdout
    t0 = time.time()
    try:
        p = subprocess.run(["mafft", "--auto", "--thread", "4", aa_in],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, timeout
    dt = time.time() - t0
    if log:
        log.write(f"$ mafft --auto {aa_in}\n# {dt:.1f}s rc={p.returncode}\n"
                  f"{p.stderr[-2000:]}\n")
    if p.returncode != 0:
        return None, dt
    open(aa_aln, "w").write(p.stdout)
    aln = read_fasta(aa_aln)
    out = {}
    for name, s in seqs.items():
        row = []
        pos = 0
        if name in bad:
            # internal stop -> sequence cannot be translation-aligned:
            # emit as all gaps (the failure mode), preserving width.
            width = len(aln[next(iter(aln))]) * 3 if aln else 0
            out[name] = "-" * width
            continue
        for a in aln[name]:
            if a == "-":
                row.append("---")
            else:
                row.append(s[pos:pos + 3])
                pos += 3
        out[name] = "".join(row)
    op = os.path.join(wd, "ts_got.fasta")
    with open(op, "w") as f:
        for name, r in out.items():
            f.write(f">{name}\n{r}\n")
    return op, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--seed-base", type=int, default=100000)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--L", type=int, default=1500)
    ap.add_argument("--gc", type=int, default=1, choices=[1, 2])
    ap.add_argument("--sub", type=float, default=0.02)
    ap.add_argument("--indel", type=float, default=0.005)
    ap.add_argument("--omega", type=float, default=0.3)
    ap.add_argument("--fs-frac", type=float, default=0.0)
    ap.add_argument("--stop-frac", type=float, default=0.0)
    ap.add_argument("--frag5", type=float, default=0.0)
    ap.add_argument("--frag3", type=float, default=0.0)
    ap.add_argument("--err", type=float, default=0.0)
    ap.add_argument("--lr-frac", type=float, default=0.0)
    ap.add_argument("--tools", default="genomsa")
    ap.add_argument("--genomsa", default="genomsa")
    ap.add_argument("--genomsa-cpu", action="store_true")
    ap.add_argument("--macse", default="macse")
    ap.add_argument("--macse-refine", action="store_true")
    ap.add_argument("--prank", default="prank")
    ap.add_argument("--prank-F", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    log = open(os.path.join(args.workdir, "run.log"), "w", buffering=1)
    seed = args.seed_base + args.rep
    sim_args = [args.n, seed, args.sub, args.indel, args.L,
                "--omega", args.omega, "--fs-frac", args.fs_frac,
                "--stop-frac", args.stop_frac, "--frag5", args.frag5,
                "--frag3", args.frag3, "--err", args.err,
                "--lr-frac", args.lr_frac, "--gc", args.gc]

    dt, rc = run([sys.executable, SIM, "-"] + [str(a) for a in sim_args],
                 600, log, cwd=args.workdir)
    if rc != 0:
        print(f"{args.cell}\t{args.rep}\tGENERATE-FAIL"); return

    inp = os.path.join(args.workdir, "sim_in.fasta")
    seqs = read_fasta(inp)
    tools = args.tools.split(",")
    rows = []

    for tool in tools:
        got = None
        dt = 0.0
        status = "ok"
        if tool == "genomsa":
            outp = os.path.join(args.workdir, "genomsa.fasta")
            cmd = [args.genomsa, inp, outp, "--codon",
                   "--gc-def", str(args.gc)]
            if args.genomsa_cpu:
                cmd.append("--cpu")
            dt, rc = run(cmd, args.timeout, log)
            if rc == 0 and os.path.exists(outp):
                got = outp
            else:
                status = f"rc={rc}"
        elif tool == "macse":
            outp = os.path.join(args.workdir, "macse_nt.fasta")
            cmd = [args.macse, "-prog", "alignSequences", "-seq", inp,
                   "-out_NT", outp]
            if not args.macse_refine:
                cmd += ["-max_refine_iter", "0"]
            if args.gc == 2:
                cmd += ["-gc_def", "2"]
            dt, rc = run(cmd, args.timeout, log)
            if rc == 0 and os.path.exists(outp):
                got = outp
            else:
                status = f"rc={rc}"
        elif tool == "prank":
            outp = os.path.join(args.workdir, "prank")
            cmd = [args.prank, f"-d={inp}", f"-o={outp}", "-codon"]
            if args.prank_F:
                cmd.append("-F")
            # prank writes <out>.best.fas
            dt, rc = run(cmd, args.timeout, log)
            cand = outp + ".best.fas"
            if rc == 0 and os.path.exists(cand):
                got = cand
            else:
                status = "no-output" if rc == 0 else f"rc={rc}"
        elif tool == "threestep":
            got, dt = threestep(seqs, args.workdir, args.gc,
                                args.timeout, log)
            if got is None:
                status = "mafft-fail"
        else:
            status = "unknown-tool"

        if got:
            sc = score(got, sim_args, args.n)
            if sc is None:
                status = "score-fail"
                rows.append([args.cell, str(args.rep), str(args.n),
                             str(args.L), str(args.gc), tool,
                             "nan", "nan", "0", "0", "0",
                             f"{dt:.0f}", status])
            else:
                rows.append([args.cell, str(args.rep), str(args.n),
                             str(args.L), str(args.gc), tool,
                             sc["sps"], sc["tc"], sc["w_got"],
                             sc["w_true"], sc["purity"],
                             f"{dt:.0f}", status])
        else:
            rows.append([args.cell, str(args.rep), str(args.n),
                         str(args.L), str(args.gc), tool,
                         "nan", "nan", "0", "0", "0",
                         f"{dt:.0f}", status])

    log.close()
    for r in rows:
        print("\t".join(r))


if __name__ == "__main__":
    main()
