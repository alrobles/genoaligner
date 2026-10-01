#!/bin/bash
#SBATCH --job-name=lf_trim_sm
#SBATCH --partition=sixhour
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#
# ClipKIT-trim the 31 genomsa --local-frame codon alignments and build the
# lf supermatrix + partitions (non-destructive: separate OUT dirs).
set -u
PAI=/beegfs/a474r867/phylogenyAI
ENV=/home/a474r867/miniconda3/envs/genoml-phylo
SRC=$PAI/data/genomsa_codon_lf
TRIM=$PAI/data/genomsa_codon_lf_trimmed
OUT=$PAI/data/supermatrix_genomsa_lf
mkdir -p "$TRIM" "$OUT"
GENES="A2AB ADORA3 ADRB2 APOB APP ATP7 BCHE BDNF BMI1 BRCA1 BRCA2 CNR1 COI
       CREM CYTB DMP1 EDG1 ENAM FBN1 GHR IRBP ND1 ND2 PLCB4 PNOC RAG1A
       RAG1B RAG2 TTN TYR1 VWF"
for g in $GENES; do
    if [ -s "$SRC/$g.codon.fasta" ] && [ ! -s "$TRIM/$g.clipkit.fasta" ]; then
        "$ENV/bin/clipkit" "$SRC/$g.codon.fasta" -m smart-gap \
            -o "$TRIM/$g.clipkit.fasta" > "$TRIM/$g.clipkit.log" 2>&1 \
            && echo "TRIM_OK $g" || echo "TRIM_FAIL $g"
    fi
done
N=$(ls "$TRIM"/*.clipkit.fasta 2>/dev/null | wc -l)
echo "trimmed: $N / 31"
[ "$N" -eq 31 ] || exit 1
TRIM_DIR=$TRIM OUT_DIR=$OUT PAI_DIR=$PAI \
    python3 /beegfs/a474r867/genoaligner/repo/scripts/pai_supermatrix.py
wc -l "$OUT/partitions.txt"
echo "SM_LF_OK"
