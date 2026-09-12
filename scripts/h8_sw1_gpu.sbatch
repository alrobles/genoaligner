#!/bin/bash -l
#SBATCH --job-name=h8_sw1
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h8sw1_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h8sw1_%j.err

# Smith-Waterman on GPU: the warp-per-row kernel.
#
# What this job establishes, stated narrowly:
#   1. CORRECTNESS at blockDim > 1 -- one warp per pair, columns tiled across
#      lanes, the intra-row dependency carried by the scan in sw_kernel.hip.
#      The CPU parity gate already ran this same body under real threaded
#      emulation at warpSize 32 AND 64; this is the check that nothing is lost
#      on a real device -- both earlier SW faults only showed on GPU.
#   2. DETERMINISM (--bench): 7 launches of identical work, reported as a
#      spread. The kernel has no barriers, so the B6 failure mode (idle lanes
#      on barriers, 2.5x between identical runs) has no mechanism here -- but
#      the criterion is measured, not assumed.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
LOGD=/beegfs/a474r867/genoaligner/logs
cd "$ROOT" || exit 1
source /kuhpc/sw/lmod/lmod/init/profile 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1

B=/beegfs/a474r867/sw_gpu
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. -Iinclude \
    -o "$B" tests/sw/test_sw_gpu.cpp || { echo "BUILD FAILED"; exit 1; }

echo "=== host: $(hostname) ==="
echo "=== SW on GPU: warp-per-row correctness + determinism ==="
echo
"$B" --bench | tee "$LOGD/sw_gpu_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
echo
echo "=== H8-SW1 RESULT ==="
if [ "$RC" -eq 0 ]; then echo "H8-SW1: PASS"; else echo "H8-SW1: FAIL (rc=$RC)"; fi
exit "$RC"
