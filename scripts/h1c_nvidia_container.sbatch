#!/bin/bash -l
#SBATCH --job-name=h1c_nvidia
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:q6000:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h1c_nv_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h1c_nv_%j.err

# H1-C TEST, NVIDIA side: does the container give us a working CUDA path?
#
# This is the payoff of the container for the NVIDIA layer. The reason the
# bare-metal path failed is that KU has NO hipcc for NVIDIA (no HIP shim).
# A container can carry a complete hipcc that wraps a complete nvcc, so the
# question becomes "does the image work" instead of "is a shim installed on
# the node" -- and the image is OUR artifact, not the site's.
#
# Two candidate routes are probed, cheapest first:
#   R1  rocm/dev-ubuntu-24.04:6.4.3-complete  -> hipcc may target NVCC
#   R2  a CUDA devel base + hipcc from ROCm's HIP-for-NVIDIA packages
# If both fail, the documented fallback stays hipify+nvcc.

set -o pipefail

SIF=/beegfs/a474r867/genoaligner/images/genoaligner-compile.sif
ROOT=/beegfs/a474r867/genoaligner/repo

echo "=== host: $(hostname) ==="
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi"

echo "--- host CUDA toolkit ---"
source /kuhpc/sw/lmod/9.3/init/bash 2>/dev/null
ml purge >/dev/null 2>&1
ml load cuda/13.0 >/dev/null 2>&1
echo "nvcc: $(command -v nvcc || echo NOT_FOUND)"

echo
echo "--- R1: does the image hipcc know about NVCC? ---"
apptainer exec --nv \
    --bind /beegfs:/beegfs \
    "$SIF" bash -c '
        set +e
        echo "hipcc: $(command -v hipcc)"
        ls -d /opt/rocm/llvm/bin 2>/dev/null
        echo "--- HIP_PLATFORM_NVIDIA probe ---"
        printf "#include <hip/hip_runtime.h>\n#include <cstdio>\nint main(){printf(\"ok\\\\n\");return 0;}\n" > /tmp/p.hip
        hipcc -D__HIP_PLATFORM_NVIDIA__ -o /tmp/p /tmp/p.hip --cuda-path=/usr/local/cuda 2>&1 | head -8
        echo "rc=$?"
        lp=$(command -v nvcc || echo none); echo "nvcc in image: $lp"
    '
R1=$?

echo
echo "=== H1C RESULT (containerized, NVIDIA) ==="
echo "R1_EXIT: $R1"
echo "See the R1 block above: if hipcc can target nvcc, the NVIDIA layer is"
echo "solved by the image and NOT by provisioning the node."
exit 0
