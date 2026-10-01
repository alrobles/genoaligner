#!/bin/bash -l
#SBATCH --job-name=rf_cmp
#SBATCH --partition=sixhour
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=01:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/logs/rf_cmp_%j.out
# Pairwise RF distances. Two groups:
#  A (full taxon set): genomsa_fixed, genomsa_prefix, macse
#  B (upham intersection, synonym-resolved): upham + each arm pruned
set -eo pipefail
source /beegfs/a474r867/miniconda3/etc/profile.d/conda.sh
conda activate genoml-phylo
R=/beegfs/a474r867/phylogenyAI/results
PAI=/beegfs/a474r867/phylogenyAI
UPH=$PAI/data/upham_ref
W=$R/rf_compare
mkdir -p "$W"
PR=/beegfs/a474r867/genoaligner/repo/scripts/upham_prune.py

rf() {  # name file
  n=$(wc -l < "$2")
  [ "$n" -ge 2 ] || { echo "== $1: skipped ($n trees)"; return; }
  echo "== $1 ($n trees)"
  iqtree2 -rf "$2" -pre "$W/$1" --quiet 2>&1 | tail -3
  grep -iE 'robinson|^ *[0-9]+ ' "$W/$1.rfdist" 2>/dev/null | head -12
}

# ---- group A: full set -------------------------------------------------
TA=$W/groupA.txt; > "$TA"; i=0
for arm in genomsa_fixed:$R/supermatrix_tree_genomsa/backbone.treefile            genomsa_prefix:$R/supermatrix_tree_genomsa/backbone_prefix.treefile            macse:$R/supermatrix_tree_macse/backbone.treefile \
            genomsa_codon:$R/supermatrix_tree_codon/backbone.treefile; do
  lbl=${arm%%:*}; f=${arm##*:}
  if [ -f "$f" ]; then cat "$f" >> "$TA"; echo "A[$i]=$lbl"; i=$((i+1)); else echo "A missing $lbl"; fi
done
rf groupA "$TA" || true

# ---- group B: upham intersection ---------------------------------------
# build upham pruned + relabeled pruned copies of each available arm
TB=$W/groupB.txt; > "$TB"; i=0
add_b() {  # label ours_tree
  [ -f "$2" ] || { echo "B missing $1"; return; }
  SMF=${3:-$PAI/data/supermatrix_genomsa/supermatrix.fasta}
  python3 "$PR" --upham "$UPH/upham_ndexp_mcc.tre"       --ours "$SMF"       --names-dmp "$PAI/data/taxonomy/names.dmp"       --ours-tree "$2" --ours-out "$W/$1.uphamset.tre"       --out "$UPH/upham_pruned_4353.tre" | tail -2
}
add_b genomsa_fixed  $R/supermatrix_tree_genomsa/backbone.treefile
add_b genomsa_prefix $R/supermatrix_tree_genomsa/backbone_prefix.treefile
add_b macse          $R/supermatrix_tree_macse/backbone.treefile
add_b genomsa_codon  $R/supermatrix_tree_codon/backbone.treefile \
                     $PAI/data/supermatrix_codon/supermatrix.fasta
if [ -f "$UPH/upham_pruned_4353.tre" ]; then
  cat "$UPH/upham_pruned_4353.tre" >> "$TB"; echo "B[$i]=upham"; i=$((i+1))
fi
for arm in genomsa_fixed genomsa_prefix macse genomsa_codon; do
  f=$W/$arm.uphamset.tre
  [ -f "$f" ] && { cat "$f" >> "$TB"; echo "B[$i]=$arm"; i=$((i+1)); }
done
rf groupB "$TB" || true
echo RF_DONE
