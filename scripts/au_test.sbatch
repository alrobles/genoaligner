#!/bin/bash -l
#SBATCH --job-name=au_test
#SBATCH --partition=sixhour
#SBATCH --exclude=r06r26n01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/logs/au_test_%j.out
#SBATCH --error=/beegfs/a474r867/phylogenyAI/logs/au_test_%j.err
# Topology test: -z candidate trees, -zb RELL bootstrap, -au AU test.
# Env: SM (supermatrix dir), SCHEME (.nex), TREES (multi-newick), PRE
set -euo pipefail
source /beegfs/a474r867/miniconda3/etc/profile.d/conda.sh
conda activate genoml-phylo

iqtree2 -s "$SM/supermatrix.fasta" -p "$SCHEME" \
    -z "$TREES" -zb 10000 -au \
    -nt 16 -ntmax 16 -pre "$PRE"
