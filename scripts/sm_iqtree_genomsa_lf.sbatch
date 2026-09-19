#!/bin/bash -l
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

# Walltime cut: reuse the merged partition scheme + models from the
# genomsa-base run (remapped to lf coordinates by remap_scheme.py), so
# ModelFinder+MERGE (~10.6h on the base run) is skipped entirely. Seed the
# search with the base backbone checkpoint tree (same taxa) via -t.
SCHEME=$SM/lf_scheme.nex
if [ ! -s "$SCHEME" ]; then
    python3 /beegfs/a474r867/genoaligner/repo/scripts/remap_scheme.py \
        "$SM/partitions.txt" \
        "$ROOT/results/supermatrix_tree_genomsa/backbone.best_scheme.nex" \
        "$SCHEME"
fi
SEED=$ROOT/results/supermatrix_tree_genomsa/backbone.treefile
[ -s "$SEED" ] || SEED=""

iqtree2 -s "$SM/supermatrix.fasta" \
    -p "$SCHEME" \
    ${SEED:+-t "$SEED"} \
    -B 1000 -T AUTO -ntmax 32 \
    -pre "$OUT/backbone" --quiet
echo "iqtree2 done"
