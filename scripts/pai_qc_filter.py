#!/usr/bin/env python3
"""Filter genes_final FASTAs using gene_qc_final TSVs.

Drop rule (conservative, corroborated):
  - always: NO_ALIGN
  - CDS-applicable genes only: tool-reported FS/STOP that is *corroborated*
    by phase-free translation of the record itself (internal in-frame stops
    in its best own frame). A clean own-frame translation overrides the
    alignment-derived flag: non-triplet indels placed at span edges or in
    UTR flanks can set fs_net without a real frameshift (seen in ND2).
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
MTDNA = {"COI","CYTB","ND1","ND2"}
STOPS = {"TAA","TAG","TGA"}
STOPS_MT = {"TAA","TAG","AGA","AGG"}

def own_internal_stops(seq, mt):
    """Min over 3 frames of internal stop count (last complete codon excluded)."""
    tab = STOPS_MT if mt else STOPS
    s = seq.upper()
    best = None
    for f in range(3):
        codons = [s[i:i+3] for i in range(f, len(s) - 2, 3)]
        n = sum(1 for c in codons[:-1] if c in tab)  # last codon = terminator
        best = n if best is None or n < best else best
    return best or 0

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
    mt = gene in MTDNA
    seqs = dict(parse_fasta(os.path.join(SRC, f"{gene}.fasta")))
    drop = {}
    n = 0
    for line in open(os.path.join(QC, fn)):
        if line.startswith("acc"): continue
        f = line.rstrip("\n").split("\t")
        # cols: acc len score si sj ei ej cov fs_runs fs_net stops stop_end flag
        acc, flag = f[0], f[12]
        fs_net, stops = int(f[9]), int(f[10])
        n += 1
        if flag == "NO_ALIGN":
            drop[acc] = "NO_ALIGN"
        elif cds_qc and (fs_net != 0 or stops > 0):
            own = own_internal_stops(seqs.get(acc, ""), mt)
            if own > 0:
                drop[acc] = "STOP" if stops > 0 else "FS"
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
