#!/bin/bash -l
#SBATCH --job-name=caster_report
#SBATCH --partition=sixhour
#SBATCH --exclude=r06r26n01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/logs/caster_report_%j.out
#SBATCH --error=/beegfs/a474r867/phylogenyAI/logs/caster_report_%j.err
set -euo pipefail

ROOT=${PHYLOGENY_ROOT:-/beegfs/a474r867/phylogenyAI}
REPO=${GENOALIGNER_REPO:-/beegfs/a474r867/genoaligner/repo}
python3 "$REPO/scripts/caster_report.py" --root "$ROOT"
