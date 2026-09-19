#!/bin/bash -l
#SBATCH --job-name=ft_sm
#SBATCH --partition=kbs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/logs/ft_sm_%j.out
#SBATCH --error=/beegfs/a474r867/phylogenyAI/logs/ft_sm_%j.err
# Quick topology proxy: FastTreeMP (GTR+CAT) on a supermatrix.
# Usage: sbatch --export=ALL,SM=<supermatrix_dir>,NAME=<tag> fasttree_sm.sbatch
set -eo pipefail
source /beegfs/a474r867/miniconda3/etc/profile.d/conda.sh
conda activate genoml-phylo
OUT=/beegfs/a474r867/phylogenyAI/results/fasttree_${NAME}
mkdir -p "$OUT"
OMP_NUM_THREADS=16 FastTreeMP -nt -gtr -gamma -spr 4 \
    -log "$OUT/ft.log" < "$SM/supermatrix.fasta" > "$OUT/ft.tre"
echo "FastTreeMP done -> $OUT/ft.tre"
