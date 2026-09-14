#!/bin/bash -l
#SBATCH --job-name=macse_iqtree
#SBATCH --partition=kbs
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --mem=128G --time=72:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/logs/macse_iqtree_%j.out
#SBATCH --error=/beegfs/a474r867/phylogenyAI/logs/macse_iqtree_%j.err
set -eo pipefail
source /beegfs/a474r867/miniconda3/etc/profile.d/conda.sh
conda activate genoml-phylo
ROOT=/beegfs/a474r867/phylogenyAI
SM=$ROOT/data/supermatrix_macse
OUT=$ROOT/results/supermatrix_tree_macse
mkdir -p "$OUT"
iqtree2 -s "$SM/supermatrix.fasta"     -p "$SM/partitions.txt"     -m MFP+MERGE -B 1000 -T AUTO -ntmax 32     -pre "$OUT/backbone" --quiet
echo "iqtree2 done"
