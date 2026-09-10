#!/bin/bash -l
#SBATCH --job-name=h1_dual_build_cuda
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:q6000:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h1_cuda_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h1_cuda_%j.err

# H1 TEST, NVIDIA side: same source, CUDA backend. Must PASS alongside the
# ROCm job for the pure-HIP strategy to be confirmed.
#
# Note: hipcc on CUDA nodes is the NVIDIA HIP shim (or ROCm's nvcc wrapper).
# If hipcc is unavailable here, the alternative is hipify + nvcc manually --
# which would itself be an important finding (HIP-pure fails => reconsider).

set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
MODROOT=/kuhpc/sw/lmod/9.3
[ -f "$MODROOT/init/bash" ] && source "$MODROOT/init/bash"

echo "=== host: $(hostname) ==="
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi"

ml purge
ml load cuda/13.0
echo "nvcc: $(command -v nvcc)"
nvcc --version 2>/dev/null | tail -2

echo "hipcc: $(command -v hipcc || echo NOT_FOUND)"

echo
echo "=== build ==="
cd "$ROOT"
if command -v hipcc >/dev/null 2>&1; then
    bash scripts/build_cuda.sh 2>&1 | tail -40
    BUILD_RC=${PIPESTATUS[0]}
else
    echo "hipcc NOT available on CUDA node -- HIP-pure build impossible here"
    BUILD_RC=2
fi

echo
echo "=== H1 RESULT (cuda) ==="
if [ "$BUILD_RC" -eq 0 ]; then
    echo "H1_CUDA: PASS"
else
    echo "H1_CUDA: FAIL (rc=$BUILD_RC)"
fi
exit "$BUILD_RC"
