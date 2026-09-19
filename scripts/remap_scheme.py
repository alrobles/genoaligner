#!/usr/bin/env python3
"""Remap an IQ-TREE best_scheme.nex onto a new supermatrix's coordinates.

Reads:
  new partitions.txt   (lines: "DNA, GENE = start-end")
  old best_scheme.nex  (charset NAME = ranges over loci; charpartition
                        mapping merged-subset name -> model)
Writes a .nex with charsets per merged subset using the NEW locus
coordinates and the OLD model per subset.

Usage: remap_scheme.py new_partitions.txt old.best_scheme.nex out.nex
"""
import re
import sys

newpart, oldscheme, outp = sys.argv[1], sys.argv[2], sys.argv[3]

# new locus ranges
locus = {}
for line in open(newpart):
    m = re.match(r"DNA,\s*(\S+)\s*=\s*(\d+)-(\d+)", line.strip())
    if m:
        locus[m.group(1)] = (int(m.group(2)), int(m.group(3)))

# old scheme: charset NAME = range ranges; name = loci joined by _
charsets = {}   # subset_name -> [loci]
models = {}
for line in open(oldscheme):
    line = line.strip()
    m = re.match(r"charset\s+(\S+)\s*=\s*(.+);", line)
    if m:
        charsets[m.group(1)] = m.group(1).split("_")
    m2 = re.match(r"(\S+):\s*([A-Za-z0-9_]+)", line)
    if m2 and m2.group(1).startswith(("GTR", "TVM", "HKY", "JC", "K80",
                                      "F81", "TN93", "TIM")):
        models[m2.group(2)] = m2.group(1)   # subset_name -> model

missing = [g for gs in charsets.values() for g in gs if g not in locus]
if missing:
    sys.exit(f"loci missing from new partitions: {missing}")

with open(outp, "w") as f:
    f.write("#nexus\nbegin sets;\n")
    for name, gs in charsets.items():
        ranges = "  ".join(f"{locus[g][0]}-{locus[g][1]}" for g in gs)
        f.write(f"  charset {name} = {ranges};\n")
    f.write("  charpartition mymodels =\n")
    items = list(charsets.items())
    for i, (name, gs) in enumerate(items):
        sep = ";" if i == len(items) - 1 else ","
        f.write(f"    {models[name]}: {name}{sep}\n")
    f.write("end;\n")
print(f"wrote {outp}: {len(charsets)} subsets, "
      f"{sum(len(g) for g in charsets.values())} loci")
