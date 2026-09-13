#!/bin/bash -l
#SBATCH --job-name=h10_sw_api
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h10_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h10_%j.err

# Smith-Waterman through the PUBLIC API (Fase C), on GPU.
#
# What this job establishes, stated narrowly:
#   1. The full CPU gate runs first (its GPU stages execute here because a
#      device is visible): the API test built with hipcc runs both WFA and SW
#      through the real device path -- buffer packing, launch contracts,
#      result fill and validators included.
#   2. EXTERNAL ORACLE AT THE API LEVEL: the TSV the API test emits via
#      --emit-sw is scored by SeqAn3 (local, affine; mapping verified by
#      seqan3_sw_oracle --selftest). Scores must agree; positions may differ
#      on ties.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
LOGD=/beegfs/a474r867/genoaligner/logs
PROBE=/beegfs/a474r867/genoaligner/seqan_probe
cd "$ROOT" || exit 1
source /kuhpc/sw/lmod/lmod/init/profile 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1

echo "=== host: $(hostname) ==="
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
/kuhpc/sw/rocm/6.4.3/bin/rocm-smi --showpids 2>/dev/null || true

export BUILD_DIR=/beegfs/a474r867/genoaligner/gate_build_h10
export HIPCC=/kuhpc/sw/rocm/6.4.3/bin/hipcc
mkdir -p "$BUILD_DIR"

echo
echo "=== full gate on a GPU node (includes API test via hipcc) ==="
bash scripts/check_kernel_cpu.sh | tee "$LOGD/h10_gate_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
[ "$RC" -ne 0 ] && { echo "=== H10: FAIL (gate rc=$RC) ==="; exit "$RC"; }

TSV="$BUILD_DIR/api_sw_gpu.tsv"
if [ ! -s "$TSV" ]; then
    echo "=== H10: FAIL (no API emit at $TSV -- the GPU API stage did not run) ==="
    exit 1
fi

echo
echo "=== external oracle: SeqAn3 on the API's own results ==="
LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64 "$PROBE/seqan3_sw_oracle" --selftest || {
    echo "=== H10: FAIL (oracle selftest) ==="; exit 1; }
LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64 \
    "$PROBE/seqan3_sw_oracle" "$TSV" "$LOGD/h10_oracle_${SLURM_JOB_ID}.tsv" || {
    echo "=== H10: FAIL (seqan3 disagrees with API results) ==="; exit 1; }

echo
echo "=== H10 RESULT ==="
echo "H10: PASS"
exit 0
