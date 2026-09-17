#!/bin/bash -l
#SBATCH --job-name=caster_cpu_scale
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/logs/caster_cpu_scale_%j.out
#SBATCH --error=/beegfs/a474r867/phylogenyAI/logs/caster_cpu_scale_%j.err
set -euo pipefail

ROOT=${PHYLOGENY_ROOT:-/beegfs/a474r867/phylogenyAI}
REPO=${GENOALIGNER_REPO:-/beegfs/a474r867/genoaligner/repo}
INPUT=${CASTER_BENCH_INPUT:-$ROOT/results/caster_backbone/benchmark/supermatrix-30000.fasta}
OUT_ROOT=${CASTER_BENCH_OUT:-$ROOT/results/caster_backbone/benchmark/cpu-scale}

for threads in 1 2 4 8 16 32; do
    THREADS=$threads CASTER_BACKEND=cpu-portable \
        "$REPO/scripts/caster_run.sh" "$INPUT" "$OUT_ROOT/t$threads"
done

python3 "$REPO/scripts/caster_scale_report.py" --out-root "$OUT_ROOT"
