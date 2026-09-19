#!/bin/bash -l
#SBATCH --job-name=codon_sm
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/logs/codon_sm_%j.out
#SBATCH --error=/beegfs/a474r867/phylogenyAI/logs/codon_sm_%j.err

# Mirror of the MACSE chain on genomsa --codon output:
#   clipkit smart-gap per locus -> pai_supermatrix -> partitions
# Output goes to data/supermatrix_codon so the MACSE chain is untouched.
set -eo pipefail
source /beegfs/a474r867/miniconda3/etc/profile.d/conda.sh
conda activate genoml-phylo
PAI=/beegfs/a474r867/phylogenyAI
SRC=$PAI/data/genomsa_codon
TRIM=$PAI/data/genomsa_codon_trimmed
OUT=$PAI/data/supermatrix_codon
mkdir -p "$TRIM" "$OUT"

for FA in "$SRC"/*.codon.fasta; do
    G=$(basename "$FA" .codon.fasta)
    clipkit "$FA" -m smart-gap -o "$TRIM/$G.clipkit.fasta" \
        > "$TRIM/$G.clipkit.log" 2>&1 || echo "CLIPKIT FAIL $G"
done
N=$(ls "$TRIM"/*.clipkit.fasta | wc -l)
echo "trimmed: $N / 31"

TRIM_DIR="$TRIM" OUT_DIR="$OUT" \
    python3 /beegfs/a474r867/genoaligner/repo/scripts/pai_supermatrix.py
wc -l "$OUT/partitions.txt"
echo "CODON_SM_OK"
