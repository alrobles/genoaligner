#!/bin/bash -l
#SBATCH --job-name=caster_mini
#SBATCH --partition=sixhour
#SBATCH --exclude=r06r26n01
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/logs/caster_mini_%A_%a.out
#SBATCH --error=/beegfs/a474r867/phylogenyAI/logs/caster_mini_%A_%a.err
set -euo pipefail

ROOT=${PHYLOGENY_ROOT:-/beegfs/a474r867/phylogenyAI}
REPO=${GENOALIGNER_REPO:-/beegfs/a474r867/genoaligner/repo}
VARIANTS=(macse base lf)
IDX=${SLURM_ARRAY_TASK_ID:?}
REP=$((IDX / ${#VARIANTS[@]}))
VARIANT=${VARIANTS[$((IDX % ${#VARIANTS[@]}))]}
INPUT=$ROOT/results/mini_backbone_v2/rep$REP/$VARIANT/supermatrix.fasta
OUT=$ROOT/results/caster_backbone/mini_v2/rep$REP/$VARIANT

THREADS=$SLURM_CPUS_PER_TASK \
    "$REPO/scripts/caster_run.sh" "$INPUT" "$OUT"
