#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: caster_run.sh INPUT_FASTA OUTPUT_DIR" >&2
    exit 2
fi

ROOT=${PHYLOGENY_ROOT:-/beegfs/a474r867/phylogenyAI}
CASTER_BIN=${CASTER_BIN:-$ROOT/tools/ASTER-v1.25/bin/caster-site-portable}
TIME_BIN=${TIME_BIN:-/usr/bin/time}
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-1}}
SEED=${SEED:-233}
CASTER_BACKEND=${CASTER_BACKEND:-cpu-portable}
INPUT=$1
OUT=$2
TREE=$OUT/caster.treefile

test -s "$INPUT"
test -x "$CASTER_BIN"
command -v flock >/dev/null
mkdir -p "$OUT"

if [ -s "$TREE" ] && grep -q ';' "$TREE"; then
    echo "CASTER output already complete: $TREE"
    exit 0
fi

exec 9> "$OUT/.caster.lock"
if ! flock -n 9; then
    echo "another CASTER run owns $OUT" >&2
    exit 75
fi
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
CASTER_SHA256=$(sha256sum "$CASTER_BIN" | awk '{print $1}')
INPUT_BYTES=$(stat -c %s "$INPUT")
HOST=$(hostname)

write_meta() {
    local status=$1
    local exit_code=$2
    local end_utc=$3
    local elapsed=$4
    local slurm_max_rss=$5
    local destination=$6
    {
        printf 'key\tvalue\n'
        printf 'status\t%s\n' "$status"
        printf 'exit_code\t%s\n' "$exit_code"
        printf 'input\t%s\n' "$INPUT"
        printf 'input_bytes\t%s\n' "$INPUT_BYTES"
        printf 'caster_bin\t%s\n' "$CASTER_BIN"
        printf 'caster_bin_sha256\t%s\n' "$CASTER_SHA256"
        printf 'caster_backend\t%s\n' "$CASTER_BACKEND"
        printf 'host\t%s\n' "$HOST"
        printf 'threads\t%s\n' "$THREADS"
        printf 'seed\t%s\n' "$SEED"
        printf 'start_utc\t%s\n' "$START_UTC"
        printf 'end_utc\t%s\n' "$end_utc"
        printf 'elapsed_seconds\t%s\n' "$elapsed"
        printf 'slurm_max_rss\t%s\n' "$slurm_max_rss"
        printf 'slurm_job_id\t%s\n' "${SLURM_JOB_ID:-local}"
        printf 'slurm_array_task_id\t%s\n' "${SLURM_ARRAY_TASK_ID:-none}"
    } > "$destination"
}

get_slurm_max_rss() {
    if [ -z "${SLURM_JOB_ID:-}" ] ||
        ! command -v sstat >/dev/null 2>&1; then
        return
    fi
    sstat -j "${SLURM_JOB_ID}.batch" \
        --format=MaxRSS --noheader --parsable2 2>/dev/null |
        awk 'NF {print $1; exit}' || true
}

write_meta running "" "" "" "" "$META_TMP"
mv "$META_TMP" "$OUT/run.meta.tsv"

finished=false
record_failure() {
    local exit_code=$1
    set +e
    if [ "$finished" = false ] && [ "$exit_code" -ne 0 ]; then
        local end_epoch
        local failure_meta
        local slurm_max_rss
        end_epoch=$(date +%s)
        failure_meta="$OUT/run.meta.tsv.failure.$$"
        slurm_max_rss=$(get_slurm_max_rss)
        write_meta failed "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            "$((end_epoch - START_EPOCH))" "$slurm_max_rss" "$failure_meta"
        mv "$failure_meta" "$OUT/run.meta.tsv"
    fi
}
trap 'record_failure "$?"' EXIT

if [ -x "$TIME_BIN" ]; then
    "$TIME_BIN" -v -o "$TIME_TMP" \
        "$CASTER_BIN" -t "$THREADS" --seed "$SEED" \
        -o "$TREE_TMP" "$INPUT" 2> "$LOG_TMP"
else
    "$CASTER_BIN" -t "$THREADS" --seed "$SEED" \
        -o "$TREE_TMP" "$INPUT" 2> "$LOG_TMP"
    printf 'GNU time unavailable on this node\n' > "$TIME_TMP"
fi

test -s "$TREE_TMP"
grep -q ';' "$TREE_TMP"

END_EPOCH=$(date +%s)
END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SLURM_MAX_RSS=$(get_slurm_max_rss)
write_meta complete 0 "$END_UTC" "$((END_EPOCH - START_EPOCH))" \
    "$SLURM_MAX_RSS" "$META_TMP"

mv "$LOG_TMP" "$OUT/caster.log"
mv "$TIME_TMP" "$OUT/caster.time"
mv "$META_TMP" "$OUT/run.meta.tsv"
mv "$TREE_TMP" "$TREE"
finished=true
trap - EXIT
echo "CASTER complete: $TREE"
