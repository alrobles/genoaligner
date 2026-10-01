#!/bin/bash -l
#SBATCH --job-name=codon_all31
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/codon31_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/codon31_%j.err

# genomsa --codon on every PhylogenyAI locus + MACSE comparison where the
# MACSE alignment already exists.
set -o pipefail

PAI=/beegfs/a474r867/phylogenyAI
REPO=${REPO:-/beegfs/a474r867/genoaligner/repo}
GENES_DIR=$PAI/data/genes_qc_pass
MACSE_DIR=$PAI/data/alignments_qc
OUT=$PAI/data/genomsa_codon
ROCM=/kuhpc/sw/rocm/6.4.3
mkdir -p "$OUT"
cd "$REPO" || exit 1

echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-local} ==="
git log --oneline -1
BIN=/tmp/codon31_${SLURM_JOB_ID:-local}; mkdir -p "$BIN"
"$ROCM/bin/hipcc" -O2 -ffp-contract=off -std=c++17 -I. -Iinclude \
    -o "$BIN/genomsa" tools/genomsa.cpp src/msa/msa_gpu.cpp \
    src/msa/nj_gpu.cpp src/msa/msa_ref.cpp || exit 1

gc_of() { case "$1" in COI|CYTB|ND1|ND2) echo 2;; *) echo 1;; esac; }

TSV=$OUT/summary_${SLURM_JOB_ID:-local}.tsv
printf "gene\tnseq\tgc\taln_len\tt_wall\tstops\tpartials\n" > "$TSV"
for FA in "$GENES_DIR"/*.fasta; do
    G=$(basename "$FA" .fasta); GC=$(gc_of "$G")
    echo "### $G gc=$GC"
    t0=$SECONDS
    if "$BIN/genomsa" "$FA" "$OUT/$G.codon.fasta" --codon --gc-def "$GC" \
            --codon-qc "$OUT/$G.qc.tsv" 2>"$OUT/$G.log"; then
        t=$((SECONDS - t0))
        NSEQ=$(grep -c '>' "$FA")
        ALEN=$(head -2 "$OUT/$G.codon.fasta" | tail -1 | tr -d '\n' | wc -c)
        ST=$(awk -F'\t' '{s+=$3} END{print s}' "$OUT/$G.qc.tsv")
        PT=$(awk -F'\t' '{s+=$4} END{print s}' "$OUT/$G.qc.tsv")
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$G" "$NSEQ" "$GC" "$ALEN" "$t" "$ST" "$PT" >> "$TSV"
        tail -1 "$OUT/$G.log"
    else
        echo "FAIL $G"; tail -3 "$OUT/$G.log"
    fi
done

echo; echo "=== vs MACSE (genes with .macse_raw_nt.fasta) ==="
for M in "$MACSE_DIR"/*.macse_raw_nt.fasta; do
    G=$(basename "$M" .macse_raw_nt.fasta)
    [ -s "$OUT/$G.codon.fasta" ] || continue
    python3 scripts/codon_vs_macse.py "$G" "$OUT/$G.codon.fasta" "$M" "$(gc_of "$G")"
done

echo; column -t "$TSV"
