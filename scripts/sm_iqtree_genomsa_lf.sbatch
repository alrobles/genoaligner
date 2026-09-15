#!/bin/bash
#SBATCH --job-name=sm_iqtree_lf
#SBATCH --partition=kbs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/logs/sm_iqtree_lf_%j.out
#SBATCH --error=/beegfs/a474r867/phylogenyAI/logs/sm_iqtree_lf_%j.err
set -eo pipefail
source /beegfs/a474r867/miniconda3/etc/profile.d/conda.sh
conda activate genoml-phylo
ROOT=/beegfs/a474r867/phylogenyAI
SM=$ROOT/data/supermatrix_genomsa_lf
OUT=$ROOT/results/supermatrix_tree_genomsa_lf
mkdir -p "$OUT"
iqtree2 -s "$SM/supermatrix.fasta" \
    -p "$SM/partitions.txt" \
    -m MFP+MERGE -B 1000 -T AUTO -ntmax 32 \
    -pre "$OUT/backbone" --quiet
echo "iqtree2 lf done"
