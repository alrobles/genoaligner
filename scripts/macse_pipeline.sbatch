#!/bin/bash -l
#SBATCH --job-name=macse_pipe
#SBATCH --partition=sixhour
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G --time=02:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/logs/macse_pipe_%j.out
#SBATCH --error=/beegfs/a474r867/phylogenyAI/logs/macse_pipe_%j.err
# MACSE downstream, mirroring genomsa_pipeline.sbatch step-for-step:
#   macse raw NT aln -> clipkit -m smart-gap -> supermatrix + partitions
set -eo pipefail
source /beegfs/a474r867/miniconda3/etc/profile.d/conda.sh
conda activate genoml-phylo

PAI=/beegfs/a474r867/phylogenyAI
SRC=$PAI/data/alignments_qc
TRIM=$PAI/data/alignments_qc_trimmed
SM=$PAI/data/supermatrix_macse
mkdir -p "$TRIM" "$SM"

N=0
for ALN in "$SRC"/*.macse_raw_nt.fasta; do
    GENE=$(basename "$ALN" .macse_raw_nt.fasta)
    clipkit "$ALN" -m smart-gap -o "$TRIM/${GENE}.clipkit.fasta"         > "$TRIM/${GENE}.clipkit.log" 2>&1         && N=$((N+1)) || echo "clipkit FAILED $GENE"
done
echo "clipkit ok: $N / 31"
[ "$N" -eq 31 ] || { echo "incomplete -- not building supermatrix"; exit 1; }

TRIM_DIR="$TRIM" OUT_DIR="$SM" PAI_DIR="$PAI"     python3 /beegfs/a474r867/genoaligner/repo/scripts/pai_supermatrix.py
NPART=$(wc -l < "$SM/partitions.txt")
[ "$NPART" -eq 31 ] || { echo "partition count $NPART != 31"; exit 1; }
echo "MACSE_PIPE_OK"
