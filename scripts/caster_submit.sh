#!/bin/bash
set -euo pipefail

ROOT=${PHYLOGENY_ROOT:-/beegfs/a474r867/phylogenyAI}
REPO=${GENOALIGNER_REPO:-/beegfs/a474r867/genoaligner/repo}
SUBMIT_USER=${USER:-$(id -un)}

if squeue -h -u "$SUBMIT_USER" -n caster_full,caster_mini | grep -q .; then
    echo "an active CASTER array already exists" >&2
    exit 75
fi

FULL_ARRAY=$(python3 "$REPO/scripts/caster_pending.py" \
    --root "$ROOT" --scope full)
MINI_ARRAY=$(python3 "$REPO/scripts/caster_pending.py" \
    --root "$ROOT" --scope mini)
JOBS=()

if [ -n "$FULL_ARRAY" ]; then
    FULL_JOB=$(sbatch --parsable --array="$FULL_ARRAY" \
        "$REPO/scripts/caster_full_array.sbatch")
    JOBS+=("$FULL_JOB")
    echo "full=$FULL_JOB array=$FULL_ARRAY"
fi
if [ -n "$MINI_ARRAY" ]; then
    MINI_JOB=$(sbatch --parsable --array="$MINI_ARRAY" \
        "$REPO/scripts/caster_mini_array.sbatch")
    JOBS+=("$MINI_JOB")
    echo "mini=$MINI_JOB array=$MINI_ARRAY"
fi

if [ "${#JOBS[@]}" -eq 0 ]; then
    REPORT_JOB=$(sbatch --parsable "$REPO/scripts/caster_report.sbatch")
else
    DEPENDENCY=$(IFS=:; echo "${JOBS[*]}")
    REPORT_JOB=$(sbatch --parsable --dependency="afterany:$DEPENDENCY" \
        "$REPO/scripts/caster_report.sbatch")
fi
echo "report=$REPORT_JOB"
