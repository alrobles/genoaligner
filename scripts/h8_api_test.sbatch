#!/bin/bash -l
#SBATCH --job-name=h8_apitest
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h8api_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h8api_%j.err

# Fase 8 — production-readiness: does the PUBLIC API actually work on a GPU?
#
# The API was written and compiled, and its shape was verified under the CPU shim
# (which cannot score through the kernel). What has never run is the real thing:
# align()/align_batch() on device, scored against the independent CPU DP.
#
# Until this passes, "the library has an API" is a compile-time claim only.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
OUT_DIR=/beegfs/a474r867/genoaligner/bench
mkdir -p "$OUT_DIR"

source /kuhpc/sw/lmod/9.3/init/bash 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1
cd "$ROOT" || exit 1

echo "=== host: $(hostname) ==="
echo "=== H8: public API on MI210 ==="
echo

/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. \
    -o /tmp/test_api_gpu tests/api/test_api.cpp src/api/api.cpp
RC_B=$?
echo "BUILD_RC=$RC_B"
[ "$RC_B" -ne 0 ] && { echo "H8: BUILD FAILED"; exit 1; }

/tmp/test_api_gpu | tee -a "$OUT_DIR/h8_api_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}

echo
echo "=== H8 RESULT ==="
if [ "$RC" -eq 0 ]; then echo "H8: PASS"; else echo "H8: FAIL (rc=$RC)"; fi
exit "$RC"
