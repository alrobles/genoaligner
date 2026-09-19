#!/bin/bash -l
#SBATCH --job-name=au_te
#SBATCH --partition=sixhour
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/beegfs/a474r867/genoaligner/tmp/au_te_%A_%a.out
#SBATCH --error=/beegfs/a474r867/genoaligner/tmp/au_te_%A_%a.err
# Fixed-topology logL evaluation (-te) for tree comparison.
# One array task per candidate tree; submit once per supermatrix via env vars:
#   sbatch --array=0-2 --export=SM=<sm.fasta>,SCHEME=<scheme.nex>,TREES=<cands.tre>,TAG=<name> au_te_array.sbatch
# Each task: iqtree2 -s $SM -p $SCHEME -te <tree_i> -> optimized logL per topology.
set -eo pipefail
ROOT=/beegfs/a474r867
IQ=/beegfs/a474r867/miniconda3/envs/genoml-phylo/bin/iqtree2
OUT=$ROOT/phylogenyAI/results/au_test
mkdir -p "$OUT"

mapfile -t CL < "$TREES"
TREE_STR="${CL[$SLURM_ARRAY_TASK_ID]}"
printf '%s\n' "$TREE_STR" > "$OUT/te_${TAG}_${SLURM_ARRAY_TASK_ID}.tre"

$IQ -s "$SM" -p "$SCHEME" \
    -te "$OUT/te_${TAG}_${SLURM_ARRAY_TASK_ID}.tre" \
    -pre "$OUT/te_${TAG}_${SLURM_ARRAY_TASK_ID}" \
    -nt "$SLURM_CPUS_PER_TASK" -seed 233
echo "TE_DONE ${TAG}_${SLURM_ARRAY_TASK_ID}"
