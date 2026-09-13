#!/bin/bash -l
#SBATCH --job-name=rf_cmp
#SBATCH --partition=sixhour
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=01:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/logs/rf_cmp_%j.out
# Pairwise RF distances among backbone trees (identical taxon set required).
# Tree order is logged to stdout; iqtree -rf indexes trees by that order.
set -eo pipefail
source /beegfs/a474r867/miniconda3/etc/profile.d/conda.sh
conda activate genoml-phylo
R=/beegfs/a474r867/phylogenyAI/results
W=$R/rf_compare
mkdir -p "$W"
TREES=$W/all_trees.txt
> "$TREES"
i=0
add() {  # label file
  if [ -f "$2" ]; then
    cat "$2" >> "$TREES"
    echo "tree $i = $1 ($2)"
    i=$((i+1))
  else
    echo "MISSING: $1 ($2)"
  fi
}
add genomsa_fixed   $R/supermatrix_tree_genomsa/backbone.treefile
add genomsa_prefix  $R/supermatrix_tree_genomsa/backbone_prefix.treefile
add macse           $R/supermatrix_tree_macse/backbone.treefile
echo "trees in file: $(wc -l < $TREES)"
[ "$i" -ge 2 ] || { echo "need >=2 trees"; exit 1; }
iqtree2 -rf "$TREES" -pre "$W/rf" --quiet 2>&1 | tail -5
cat "$W/rf.rfdist" 2>/dev/null || cat "$W/rf.iqtree" 2>/dev/null | grep -A20 -i robinson
echo RF_DONE
