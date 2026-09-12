#!/bin/bash -l
#SBATCH --job-name=h8_b2
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h8b2_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h8b2_%j.err

# Production flag B2 — alignment on REAL biological sequence.
#
# Every other correctness test in this project generates its input. This one runs on
# real mtDNA (human NC_012920.1, chimp NC_001643.1) because real genomes contain
# repeats, low-complexity and structured regions that a random generator does not
# produce, and that is where traceback implementations break.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
OUT_DIR=/beegfs/a474r867/genoaligner/bench
mkdir -p "$OUT_DIR"

source /kuhpc/sw/lmod/lmod/init/profile 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1
cd "$ROOT" || exit 1

echo "=== host: $(hostname) ==="
echo "=== H8-B2: real-sequence alignment on MI210 ==="
echo

/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. \
    -o /tmp/test_real tests/real/test_real_sequences.cpp src/api/api.cpp
RC_B=$?
echo "BUILD_RC=$RC_B"
[ "$RC_B" -ne 0 ] && { echo "H8-B2: BUILD FAILED"; exit 1; }

/tmp/test_real | tee -a "$OUT_DIR/h8b2_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}

echo
echo "=== H8-B2 RESULT ==="
if [ "$RC" -eq 0 ]; then echo "H8-B2: PASS"; else echo "H8-B2: FAIL (rc=$RC)"; fi
exit "$RC"
