#!/bin/bash -l
#SBATCH --job-name=caster_full
#SBATCH --partition=kbs
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/logs/caster_full_%A_%a.out
#SBATCH --error=/beegfs/a474r867/phylogenyAI/logs/caster_full_%A_%a.err
set -euo pipefail

ROOT=${PHYLOGENY_ROOT:-/beegfs/a474r867/phylogenyAI}
REPO=${GENOALIGNER_REPO:-/beegfs/a474r867/genoaligner/repo}
VARIANTS=(genomsa genomsa_lf codon)
INPUTS=(
    "$ROOT/data/supermatrix_genomsa/supermatrix.fasta"
    "$ROOT/data/supermatrix_genomsa_lf/supermatrix.fasta"
    "$ROOT/data/supermatrix_codon/supermatrix.fasta"
)

IDX=${SLURM_ARRAY_TASK_ID:?}
VARIANT=${VARIANTS[$IDX]}
INPUT=${INPUTS[$IDX]}
OUT=$ROOT/results/caster_backbone/full/$VARIANT

THREADS=$SLURM_CPUS_PER_TASK \
    "$REPO/scripts/caster_run.sh" "$INPUT" "$OUT"
