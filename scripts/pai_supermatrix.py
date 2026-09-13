#!/usr/bin/env python3
"""Concatenate per-gene trimmed alignments into a supermatrix + partitions.

Taxon key = species binomial parsed from genes_raw GenBank headers
("ACC.V Species name isolate ..." -> Species_name). Accessions not found
in genes_raw fall back to the accession itself. When several accessions of
the same species occur in one gene, the copy with the fewest gap/N/? sites
is kept (reported in dup_report.tsv).

Reads  data/alignments_qc_trimmed/<GENE>.clipkit.fasta
Writes data/supermatrix/supermatrix.fasta
       data/supermatrix/partitions.txt  (RAxML/IQ-TREE: DNA, GENE=a-b)
       data/supermatrix/taxon_coverage.tsv
       data/supermatrix/dup_report.tsv
"""
import os, sys, re

PAI = os.environ.get("PAI_DIR", "/beegfs/a474r867/phylogenyAI")
RAW = os.path.join(PAI, "data/genes_raw")
TRIM = os.environ.get("TRIM_DIR", os.path.join(PAI, "data/alignments_qc_trimmed"))
OUT = os.environ.get("OUT_DIR", os.path.join(PAI, "data/supermatrix"))
GENES = ["A2AB","ADORA3","ADRB2","APOB","APP","ATP7","BCHE","BDNF","BMI1",
         "BRCA1","BRCA2","CNR1","COI","CREM","CYTB","DMP1","EDG1","ENAM",
         "FBN1","GHR","IRBP","ND1","ND2","PLCB4","PNOC","RAG1A","RAG1B",
         "RAG2","TTN","TYR1","VWF"]
SKIP_TOKENS = {"sp.","cf.","aff.","x","hybrid"}

def parse_fasta(path):
    for blk in open(path).read().split(">"):
        if not blk.strip(): continue
        h, *ls = blk.split("\n")
        yield h.strip(), "".join(ls).strip()

def species_of(header):
    """'ACC.V Species epithet isolate X ...' -> 'Species_epithet'"""
    toks = header.split()
    if len(toks) < 3: return None
    g, s = toks[1], toks[2]
    if s.lower() in SKIP_TOKENS or not g[0].isupper(): return None
    return re.sub(r"[^A-Za-z_]", "", g) + "_" + re.sub(r"[^A-Za-z]", "", s)

# accession -> species, built once from raw headers across all genes
acc2sp = {}
for g in GENES:
    p = os.path.join(RAW, f"{g}.fasta")
    if not os.path.exists(p): continue
    for h, _ in parse_fasta(p):
        acc = h.split()[0]
        acc2sp.setdefault(acc, species_of(h))

os.makedirs(OUT, exist_ok=True)
matrix = {}          # taxon -> {gene: seq}
parts = []           # (gene, start, end) 1-based
dup_rows = []
pos = 1
for g in GENES:
    p = os.path.join(TRIM, f"{g}.clipkit.fasta")
    if not os.path.exists(p):
        print(f"WARN missing {p}", file=sys.stderr); continue
    aln = list(parse_fasta(p))
    L = len(aln[0][1])
    assert all(len(s) == L for _, s in aln), f"{g}: ragged alignment"
    best = {}
    for h, s in aln:
        acc = h.split()[0]
        tx = acc2sp.get(acc) or acc
        bad = sum(1 for c in s if c in "-N?n")
        if tx not in best or bad < best[tx][0]:
            if tx in best: dup_rows.append((g, tx, "replaced"))
            best[tx] = (bad, s)
    for tx, (bad, s) in best.items():
        matrix.setdefault(tx, {})[g] = s
    parts.append((g, pos, pos + L - 1))
    pos += L
    print(f"{g:8s} {len(aln):5d} seqs -> {len(best):5d} taxa, L={L}")

total = pos - 1
with open(os.path.join(OUT, "supermatrix.fasta"), "w") as o:
    for tx in sorted(matrix):
        row = "".join(matrix[tx].get(g, "-" * (e - s + 1)) for g, s, e in parts)
        assert len(row) == total
        o.write(f">{tx}\n{row}\n")
with open(os.path.join(OUT, "partitions.txt"), "w") as o:
    for g, s, e in parts:
        o.write(f"DNA, {g} = {s}-{e}\n")
with open(os.path.join(OUT, "taxon_coverage.tsv"), "w") as o:
    o.write("taxon\tgenes_present\n")
    for tx in sorted(matrix):
        o.write(f"{tx}\t{len(matrix[tx])}\n")
with open(os.path.join(OUT, "dup_report.tsv"), "w") as o:
    o.write("gene\ttaxon\tnote\n")
    for r in dup_rows: o.write("\t".join(r) + "\n")
print(f"supermatrix: {len(matrix)} taxa x {total} sites, {len(parts)} partitions")
