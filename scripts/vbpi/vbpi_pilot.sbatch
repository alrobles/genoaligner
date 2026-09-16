#!/bin/bash -l
#SBATCH --job-name=vbpi_pilot
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:mi210:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=r06r26n01
#SBATCH --output=/beegfs/a474r867/genoaligner/tmp/vbpi_pilot_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/tmp/vbpi_pilot_%j.err
# VBPI scaling pilot on the lf supermatrix: n in {128,256,512,1024}.
# Measures iter time / GPU mem to decide if n=4353 is feasible.
set -eo pipefail
ROOT=/beegfs/a474r867
VBPI=$ROOT/vbpi-torch/unrooted
REPO=$ROOT/genoaligner/repo/scripts/vbpi
WORK=$ROOT/phylogenyAI/results/vbpi_pilot
SM=$ROOT/phylogenyAI/data/supermatrix_genomsa_lf/supermatrix.fasta
mkdir -p "$WORK"

# patch upstream clone once (idempotent)
[ -f "$VBPI/.patched" ] || { python3 $REPO/patch_vbpi_torch.py "$VBPI" && touch "$VBPI/.patched"; }

# support trees: every available backbone checkpoint + ufboot + Upham ref
TREES=""
for f in $ROOT/phylogenyAI/results/*/backbone.treefile \
         $ROOT/phylogenyAI/results/*/backbone.ufboot \
         $ROOT/phylogenyAI/data/upham_ref/upham_pruned_4353.tre; do
  [ -f "$f" ] && TREES="$TREES $f"
done
[ -n "$TREES" ] || { echo "no support trees found"; exit 1; }
echo "support sources:$TREES"

for N in 128 256 512 1024; do
  D=$WORK/n$N
  if [ ! -s "$D/support.trees" ]; then
    /beegfs/a474r867/miniconda3/envs/vbpi/bin/python $REPO/prep_pilot.py \
        --fasta $SM --n $N --seed 0 --trees $TREES --outdir $D
  fi
  /beegfs/a474r867/miniconda3/envs/vbpi/bin/python $REPO/vbpi_driver.py \
      --vbpi-src $VBPI --fasta $D/sub.fasta --trees $D/support.trees \
      --out $D/vbpi --maxIter 2000 --nParticle 10 --tf 200 \
      --nwarmStart 500 --device cuda 2>&1 | tail -25
  echo "=== n=$N done ==="
done
echo VBPI_PILOT_DONE
