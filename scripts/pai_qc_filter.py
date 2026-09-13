#!/usr/bin/env python3
"""Filter genes_final FASTAs using gene_qc_final TSVs.

Drop rule (conservative):
  - always: NO_ALIGN
  - CDS-applicable genes only: hard frameshift (fs_net != 0) or internal stops
  - LOW_COV kept (legitimate fragments; triage downstream)

Genes whose locus contains introns/UTR (no same-locus record translates
cleanly) get homology-only QC; their stop/FS signals are structurally
meaningless and cleaning defers to gene-tree inspection.
Writes genes_qc_pass/<gene>.fasta + drop_list.tsv + summary.tsv.
"""
import os, sys

PAI = os.environ.get("PAI_DIR", "/beegfs/a474r867/phylogenyAI")
QC = os.path.join(PAI, "data/gene_qc_final")
SRC = os.path.join(PAI, "data/genes_final")
OUT = os.path.join(PAI, "data/genes_qc_pass")

# loci where no same-locus record is stop-free (bait region not pure CDS)
NON_CDS_LOCI = {"APP","BMI1","CREM","FBN1","GHR","ND1","PLCB4","TYR1","PNOC"}

def parse_fasta(path):
    seqs, cur, name = [], [], None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name is not None: seqs.append((name, "".join(cur)))
            name, cur = line[1:].split()[0], []
        else:
            cur.append(line.strip())
    if name is not None: seqs.append((name, "".join(cur)))
    return seqs

os.makedirs(OUT, exist_ok=True)
summary = []
for fn in sorted(os.listdir(QC)):
    if not fn.endswith(".qc.tsv"): continue
    gene = fn.split(".")[0]
    cds_qc = gene not in NON_CDS_LOCI
    drop = {}
    n = 0
    for line in open(os.path.join(QC, fn)):
        if line.startswith("acc"): continue
        f = line.rstrip("\n").split("\t")
        acc, flag = f[0], f[12]
        fs_net, stops = int(f[10]), int(f[11])
        n += 1
        if flag == "NO_ALIGN":
            drop[acc] = "NO_ALIGN"
        elif cds_qc:
            if fs_net != 0:
                drop[acc] = "FS"
            elif stops > 0:
                drop[acc] = "STOP"
    src = parse_fasta(os.path.join(SRC, f"{gene}.fasta"))
    kept = [(a, s) for a, s in src if a not in drop]
    missing = set(drop) - {a for a, _ in src}
    if missing:
        print(f"WARN {gene}: {len(missing)} dropped accs not in fasta", file=sys.stderr)
    with open(os.path.join(OUT, f"{gene}.fasta"), "w") as o:
        for a, s in kept:
            o.write(f">{a}\n{s}\n")
    with open(os.path.join(OUT, f"{gene}.drop.tsv"), "w") as o:
        o.write("acc\treason\n")
        for a, r in sorted(drop.items()):
            o.write(f"{a}\t{r}\n")
    summary.append((gene, n, len(kept), len(drop), cds_qc))

with open(os.path.join(OUT, "summary.tsv"), "w") as o:
    o.write("gene\trecords\tkept\tdropped\tcds_qc\n")
    for g, n, k, d, c in summary:
        o.write(f"{g}\t{n}\t{k}\t{d}\t{c}\n")
for g, n, k, d, c in summary:
    print(f"{g:8s} {n:6d} -> kept {k:6d} dropped {d:5d} cds_qc={c}")
print(f"total kept {sum(s[2] for s in summary)} / {sum(s[1] for s in summary)}")
