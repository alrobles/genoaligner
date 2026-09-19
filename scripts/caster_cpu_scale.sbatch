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
SOURCE=${CASTER_BENCH_SOURCE:-$ROOT/data/supermatrix_genomsa/supermatrix.fasta}
SITES=${CASTER_BENCH_SITES:-3000}
REQUESTED_THREADS=${CASTER_BENCH_THREADS:-}
CASTER_CHUNK=${CASTER_BENCH_CHUNK:-10000}
INPUT=${CASTER_BENCH_INPUT:-$ROOT/results/caster_backbone/benchmark/supermatrix-${SITES}.fasta}
OUT_ROOT=${CASTER_BENCH_OUT:-$ROOT/results/caster_backbone/benchmark/cpu-scale-${SITES}}

if [ ! -s "$INPUT" ]; then
    python3 "$REPO/scripts/caster_sample_alignment.py" \
        --input "$SOURCE" --output "$INPUT" --sites "$SITES"
fi

PLAN_ARGS=(
    --input "$INPUT"
    --chunk "$CASTER_CHUNK"
    --max-threads "${SLURM_CPUS_PER_TASK:-32}"
)
if [ -n "$REQUESTED_THREADS" ]; then
    PLAN_ARGS+=(--threads "$REQUESTED_THREADS")
fi
IFS=$'\t' read -r ALIGNMENT_SITES EFFECTIVE_CHUNKS THREAD_COUNTS < <(
    python3 "$REPO/scripts/caster_thread_plan.py" "${PLAN_ARGS[@]}"
)
echo "alignment_sites=$ALIGNMENT_SITES"
echo "caster_chunk=$CASTER_CHUNK"
echo "effective_scoring_chunks=$EFFECTIVE_CHUNKS"
echo "thread_counts=$THREAD_COUNTS"

for threads in $THREAD_COUNTS; do
    THREADS=$threads CASTER_CHUNK=$CASTER_CHUNK \
        CASTER_BACKEND=cpu-portable \
        "$REPO/scripts/caster_run.sh" "$INPUT" "$OUT_ROOT/t$threads"
done

THREAD_CSV=$(echo "$THREAD_COUNTS" | xargs | tr ' ' ',')
python3 "$REPO/scripts/caster_scale_report.py" \
    --out-root "$OUT_ROOT" --threads "$THREAD_CSV"
