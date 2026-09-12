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

# Smith-Waterman, step 1: is the kernel CORRECT on a GPU, in the one configuration the
# gate can vouch for (blockDim=1, sequential row walk)?
#
# This is deliberately the smallest claim worth making. The kernel passes the CPU shim
# gate against the independent full-matrix reference at blockDim=1, but the shim's
# __syncthreads is a no-op, so a GPU run is the first time the shared-memory row walk
# executes with real barriers at all.
#
# blockDim > 1 is NOT tested here: the column 1 stride reads H[j-1]/E[j-1] written by
# another thread in the same row, and one barrier per row does not make that visible.
# That is a known defect, recorded in the kernel's own gate output. Until it is fixed,
# the launch uses 1 thread per block, which is correct but slow -- and slow-but-correct
# is the honest place to start.
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
echo "=== SW on GPU: correctness of the sequential row walk ==="
echo
"$B" | tee "$LOGD/sw_gpu_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
echo
echo "=== H8-SW1 RESULT ==="
if [ "$RC" -eq 0 ]; then echo "H8-SW1: PASS"; else echo "H8-SW1: FAIL (rc=$RC)"; fi
exit "$RC"
