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
CASTER_CHUNK=${CASTER_CHUNK:-10000}
INPUT=$1
OUT=$2
TREE=$OUT/caster.treefile

command -v flock >/dev/null
if ! [[ "$CASTER_CHUNK" =~ ^[1-9][0-9]*$ ]]; then
    echo "CASTER_CHUNK must be a positive integer" >&2
    exit 2
fi

mkdir -p "$OUT"
exec 9> "$OUT/.caster.lock"
if ! flock -n 9; then
    echo "another CASTER run owns $OUT" >&2
    exit 75
fi

BUILD_META="${CASTER_BIN}.build.tsv"
read_tsv_value() {
    if [ -s "$1" ]; then
        awk -F '\t' -v key="$2" '$1 == key {print $2; exit}' "$1"
    fi
}

START_RT=$(date +%s.%N)
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TREE_TMP=$OUT/caster.treefile.tmp
LOG_TMP=$OUT/caster.log.tmp
TIME_TMP=$OUT/caster.time.tmp
META_TMP=$OUT/run.meta.tsv.tmp

INPUT_BYTES=""
INPUT_SHA256=""
CASTER_SHA256=""
CASTER_BUILD_PROFILE=""
CASTER_ASTER_COMMIT=""
CASTER_COMPILER=""
CASTER_FLAGS=""
CASTER_CONFIG_SHA256=""
DECLARED_BACKEND=$(read_tsv_value "$BUILD_META" runtime_backend)
CASTER_BACKEND=${CASTER_BACKEND:-${DECLARED_BACKEND:-cpu-unknown}}
HOST=$(hostname)
HOST_ARCH=$(uname -m)

elapsed_seconds() {
    awk -v start="$START_RT" -v end="$(date +%s.%N)" \
        'BEGIN {printf "%.6f", end - start}'
}

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
        printf 'input_sha256\t%s\n' "$INPUT_SHA256"
        printf 'caster_bin\t%s\n' "$CASTER_BIN"
        printf 'caster_bin_sha256\t%s\n' "$CASTER_SHA256"
        printf 'caster_backend\t%s\n' "$CASTER_BACKEND"
        printf 'caster_build_profile\t%s\n' "$CASTER_BUILD_PROFILE"
        printf 'caster_aster_commit\t%s\n' "$CASTER_ASTER_COMMIT"
        printf 'caster_compiler\t%s\n' "$CASTER_COMPILER"
        printf 'caster_flags\t%s\n' "$CASTER_FLAGS"
        printf 'caster_config_sha256\t%s\n' "$CASTER_CONFIG_SHA256"
        printf 'host\t%s\n' "$HOST"
        printf 'host_arch\t%s\n' "$HOST_ARCH"
        printf 'threads\t%s\n' "$THREADS"
        printf 'seed\t%s\n' "$SEED"
        printf 'chunk\t%s\n' "$CASTER_CHUNK"
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

finish_early() {
    local status=$1
    local exit_code=$2
    local message=$3
    write_meta "$status" "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$(elapsed_seconds)" "" "$META_TMP"
    mv "$META_TMP" "$OUT/run.meta.tsv"
    echo "$message" >&2
    exit "$exit_code"
}

if [ ! -s "$INPUT" ] || [ ! -r "$INPUT" ]; then
    finish_early invalid_input 66 "input is missing or empty: $INPUT"
fi
if [ ! -x "$CASTER_BIN" ]; then
    finish_early invalid_input 69 \
        "caster binary is not executable: $CASTER_BIN"
fi
if [ -n "$DECLARED_BACKEND" ] && [ "$CASTER_BACKEND" != "$DECLARED_BACKEND" ]; then
    finish_early invalid_input 65 \
        "CASTER_BACKEND=$CASTER_BACKEND does not match $DECLARED_BACKEND"
fi

INPUT_BYTES=$(stat -c %s "$INPUT")
INPUT_SHA256=$(sha256sum "$INPUT" | awk '{print $1}')
CASTER_SHA256=$(sha256sum "$CASTER_BIN" | awk '{print $1}')
CASTER_BUILD_PROFILE=$(read_tsv_value "$BUILD_META" profile)
CASTER_ASTER_COMMIT=$(read_tsv_value "$BUILD_META" aster_commit)
CASTER_COMPILER=$(read_tsv_value "$BUILD_META" compiler)
CASTER_FLAGS=$(read_tsv_value "$BUILD_META" flags)
CASTER_CONFIG_SHA256=$(
    printf '%s\n' \
        "$INPUT_SHA256" "$CASTER_SHA256" "$CASTER_BACKEND" \
        "$CASTER_BUILD_PROFILE" "$CASTER_ASTER_COMMIT" \
        "$CASTER_COMPILER" "$CASTER_FLAGS" "$THREADS" "$SEED" \
        "$CASTER_CHUNK" |
        sha256sum | awk '{print $1}'
)

if [ -s "$TREE" ] && grep -q ';' "$TREE"; then
    previous_status=$(read_tsv_value "$OUT/run.meta.tsv" status)
    previous_config=$(read_tsv_value "$OUT/run.meta.tsv" caster_config_sha256)
    if [ "$previous_status" = "complete" ] &&
        [ -n "$previous_config" ] &&
        [ "$previous_config" = "$CASTER_CONFIG_SHA256" ]; then
        echo "CASTER output already complete: $TREE"
        exit 0
    fi
    # A tree without a matching manifest cannot be trusted as this run's
    # output; archive it instead of reusing or silently overwriting.
    stamp=$(date +%s%N)
    for artifact in caster.treefile run.meta.tsv caster.log caster.time; do
        if [ -e "$OUT/$artifact" ]; then
            mv "$OUT/$artifact" "$OUT/$artifact.stale.$stamp"
        fi
    done
fi

write_meta running "" "" "" "" "$META_TMP"
mv "$META_TMP" "$OUT/run.meta.tsv"

finished=false
record_failure() {
    local exit_code=$1
    set +e
    if [ "$finished" = false ] && [ "$exit_code" -ne 0 ]; then
        local status=failed
        if [ "$exit_code" -eq 143 ]; then
            status=timeout
        fi
        local failure_meta
        local slurm_max_rss
        failure_meta="$OUT/run.meta.tsv.failure.$$"
        slurm_max_rss=$(get_slurm_max_rss)
        write_meta "$status" "$exit_code" \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            "$(elapsed_seconds)" "$slurm_max_rss" "$failure_meta"
        mv "$failure_meta" "$OUT/run.meta.tsv"
        if [ -e "$LOG_TMP" ]; then
            mv "$LOG_TMP" "$OUT/caster.log"
        fi
        if [ -e "$TIME_TMP" ]; then
            mv "$TIME_TMP" "$OUT/caster.time"
        fi
        rm -f "$TREE_TMP"
    fi
}
trap 'record_failure "$?"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

if [ -x "$TIME_BIN" ]; then
    "$TIME_BIN" -v -o "$TIME_TMP" \
        "$CASTER_BIN" -t "$THREADS" --seed "$SEED" \
        --chunk "$CASTER_CHUNK" \
        -o "$TREE_TMP" "$INPUT" 2> "$LOG_TMP"
else
    "$CASTER_BIN" -t "$THREADS" --seed "$SEED" \
        --chunk "$CASTER_CHUNK" \
        -o "$TREE_TMP" "$INPUT" 2> "$LOG_TMP"
    printf 'GNU time unavailable on this node\n' > "$TIME_TMP"
fi

test -s "$TREE_TMP"
grep -q ';' "$TREE_TMP"

END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SLURM_MAX_RSS=$(get_slurm_max_rss)
write_meta complete 0 "$END_UTC" "$(elapsed_seconds)" \
    "$SLURM_MAX_RSS" "$META_TMP"

mv "$LOG_TMP" "$OUT/caster.log"
mv "$TIME_TMP" "$OUT/caster.time"
mv "$META_TMP" "$OUT/run.meta.tsv"
mv "$TREE_TMP" "$TREE"
finished=true
trap - EXIT TERM INT HUP
echo "CASTER complete: $TREE"
