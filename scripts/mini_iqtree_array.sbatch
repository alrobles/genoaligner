#!/bin/bash -l
#SBATCH --job-name=mini_iqt
#SBATCH --partition=sixhour
#SBATCH --exclude=r06r26n01
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/logs/mini_iqt_%A_%a.out
#SBATCH --error=/beegfs/a474r867/phylogenyAI/logs/mini_iqt_%A_%a.err
# Mini-backbone array: task = rep*3 + variant_idx
# Usage: sbatch --array=0-14 --export=ALL,MINI=<mini dir>,VARIANTS="macse base lf" mini_iqtree_array.sbatch
set -eo pipefail
source /beegfs/a474r867/miniconda3/etc/profile.d/conda.sh
conda activate genoml-phylo
read -ra V <<< "$VARIANTS"
REP=$((SLURM_ARRAY_TASK_ID / ${#V[@]}))
VI=$((SLURM_ARRAY_TASK_ID % ${#V[@]}))
VAR=${V[$VI]}
D=$MINI/rep$REP/$VAR
mkdir -p "$D/tree"
iqtree2 -s "$D/supermatrix.fasta" -p "$D/partitions.txt" \
    -m MFP+MERGE -B 1000 -T 8 -ntmax 8 -seed 42 \
    -pre "$D/tree/backbone" --quiet
echo "done rep=$REP var=$VAR"
