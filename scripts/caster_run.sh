#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: caster_run.sh INPUT_FASTA OUTPUT_DIR" >&2
    exit 2
fi

ROOT=${PHYLOGENY_ROOT:-/beegfs/a474r867/phylogenyAI}
CASTER_BIN=${CASTER_BIN:-$ROOT/tools/ASTER-v1.25/bin/caster-site}
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-1}}
SEED=${SEED:-233}
INPUT=$1
OUT=$2
TREE=$OUT/caster.treefile

test -s "$INPUT"
test -x "$CASTER_BIN"
mkdir -p "$OUT"

if [ -s "$TREE" ] && grep -q ';' "$TREE"; then
    echo "CASTER output already complete: $TREE"
    exit 0
fi

START_EPOCH=$(date +%s)
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TREE_TMP=$OUT/caster.treefile.tmp
LOG_TMP=$OUT/caster.log.tmp
TIME_TMP=$OUT/caster.time.tmp
META_TMP=$OUT/run.meta.tsv.tmp

/usr/bin/time -v -o "$TIME_TMP" \
    "$CASTER_BIN" -t "$THREADS" --seed "$SEED" \
    -o "$TREE_TMP" "$INPUT" 2> "$LOG_TMP"

test -s "$TREE_TMP"
grep -q ';' "$TREE_TMP"

END_EPOCH=$(date +%s)
END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
    printf 'key\tvalue\n'
    printf 'status\tcomplete\n'
    printf 'input\t%s\n' "$INPUT"
    printf 'caster_bin\t%s\n' "$CASTER_BIN"
    printf 'threads\t%s\n' "$THREADS"
    printf 'seed\t%s\n' "$SEED"
    printf 'start_utc\t%s\n' "$START_UTC"
    printf 'end_utc\t%s\n' "$END_UTC"
    printf 'elapsed_seconds\t%s\n' "$((END_EPOCH - START_EPOCH))"
    printf 'slurm_job_id\t%s\n' "${SLURM_JOB_ID:-local}"
    printf 'slurm_array_task_id\t%s\n' "${SLURM_ARRAY_TASK_ID:-none}"
} > "$META_TMP"

mv "$LOG_TMP" "$OUT/caster.log"
mv "$TIME_TMP" "$OUT/caster.time"
mv "$META_TMP" "$OUT/run.meta.tsv"
mv "$TREE_TMP" "$TREE"
echo "CASTER complete: $TREE"
