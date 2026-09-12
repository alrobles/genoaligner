#!/bin/bash -l
#SBATCH --job-name=h8_b3b4
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h8b3_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h8b3_%j.err

# Production flags B3 (concurrency) and B4 (batch semantics).
#
# B3: the header claims a thread-safety contract. This job is what makes that claim
# true rather than aspirational -- the comment was written before the measurement,
# which is exactly the order this project has learned to avoid.
#
# B4: pins the documented batch semantics (one smax for the whole batch, the MINIMUM
# of the requests) by exercising a mixed-bound batch and checking the observable
# consequence: the narrowest request governs.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
OUT_DIR=/beegfs/a474r867/genoaligner/bench
mkdir -p "$OUT_DIR"

source /kuhpc/sw/lmod/lmod/init/profile 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1
cd "$ROOT" || exit 1

echo "=== host: $(hostname) ==="
echo "=== H8-B3/B4: concurrency and batch semantics on MI210 ==="
echo

/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -pthread -I. \
    -o /tmp/test_conc tests/concurrency/test_concurrency.cpp src/api/api.cpp
RC_B=$?
echo "BUILD_RC=$RC_B"
[ "$RC_B" -ne 0 ] && { echo "H8-B3: BUILD FAILED"; exit 1; }

/tmp/test_conc | tee "$OUT_DIR/h8b3_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}

echo
echo "=== H8-B3/B4 RESULT ==="
if [ "$RC" -eq 0 ]; then echo "H8-B3/B4: PASS"; else echo "H8-B3/B4: FAIL (rc=$RC)"; fi
exit "$RC"
