#!/bin/bash -l
#SBATCH --job-name=post_chain_rf
#SBATCH --partition=sixhour
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/beegfs/a474r867/genoaligner/tmp/post_chain_rf_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/tmp/post_chain_rf_%j.err
# Runs after the 4 lf chains: picks best-logL chain tree, computes
# RF vs Upham, appends a durable row to results/post_chain_rf.tsv.
#   sbatch --dependency=afterany:<j1>:<j2>:<j3>:<j4> post_chain_rf.sbatch
set -eo pipefail
ROOT=/beegfs/a474r867
RES=$ROOT/phylogenyAI/results/supermatrix_tree_genomsa_lf
UPHAM=$ROOT/phylogenyAI/data/upham_ref/upham_pruned_4353.tre
RF=$ROOT/genoaligner/repo/scripts/rf_distance.py
OUT=$ROOT/phylogenyAI/results/post_chain_rf.tsv
mkdir -p "$(dirname "$OUT")"

best_logl=-inf; best_tree=''
for s in 1 2 3 4; do
  log="$RES/backbone_s${s}.log"
  tree="$RES/backbone_s${s}.treefile"
  [ -f "$tree" ] || continue
  l=$(grep -oE 'BETTER TREE FOUND at iteration [0-9]+: -?[0-9.]+' "$log" \
      | tail -1 | grep -oE '\-?[0-9.]+$')
  [ -z "$l" ] && l=$(grep -oE 'Log-likelihood of the tree: -?[0-9.]+' "$log" \
      | tail -1 | grep -oE '\-?[0-9.]+$')
  echo "s$s logL=$l tree=$tree"
  if [ -n "$l" ] && awk "BEGIN{exit !($l>$best_logl)}"; then
    best_logl=$l; best_tree=$tree
  fi
done
echo "BEST seed_tree=$best_tree logL=$best_logl"

{
  echo -e "# $(date -Iseconds)"
  for s in 1 2 3 4; do
    t="$RES/backbone_s${s}.treefile"
    [ -f "$t" ] && echo -e "lf_chain_s${s}\t$(python3 "$RF" "$UPHAM" "$t")"
  done
  echo -e "lf_chain_best\t$(python3 "$RF" "$UPHAM" "$best_tree")"
  echo -e "lf_chain_best_logl\t$best_logl"
} >> "$OUT"
echo "POST_CHAIN_RF_DONE -> $OUT"
