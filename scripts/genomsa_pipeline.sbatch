#!/bin/bash -l
#SBATCH --job-name=genomsa_pipe
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/gsm_pipe_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/gsm_pipe_%j.err

# genomsa downstream pipeline, mirroring the MACSE one step-for-step:
#   genomsa aln -> clipkit -m smart-gap -> supermatrix + partitions
# IQ-TREE is a separate dependent job (same script as MACSE, other dirs).
set -eo pipefail
source /beegfs/a474r867/miniconda3/etc/profile.d/conda.sh
conda activate genoml-phylo

PAI=/beegfs/a474r867/phylogenyAI
SRC=$PAI/data/msa_genomsa
TRIM=$PAI/data/msa_genomsa_trimmed
SM=$PAI/data/supermatrix_genomsa
mkdir -p "$TRIM" "$SM"

N=0
for ALN in "$SRC"/*.genomsa.aln.fasta; do
    GENE=$(basename "$ALN" .genomsa.aln.fasta)
    clipkit "$ALN" -m smart-gap -o "$TRIM/${GENE}.clipkit.fasta" \
        > "$TRIM/${GENE}.clipkit.log" 2>&1 \
        && N=$((N+1)) || echo "clipkit FAILED $GENE"
done
echo "clipkit ok: $N / 31"
[ "$N" -eq 31 ] || { echo "incomplete -- not building supermatrix"; exit 1; }

TRIM_DIR="$TRIM" OUT_DIR="$SM" PAI_DIR="$PAI" \
    python3 /beegfs/a474r867/genoaligner/repo/scripts/pai_supermatrix.py
NPART=$(wc -l < "$SM/partitions.txt")
[ "$NPART" -eq 31 ] || { echo "partition count $NPART != 31"; exit 1; }
echo "GENOMSA_PIPE_OK"
