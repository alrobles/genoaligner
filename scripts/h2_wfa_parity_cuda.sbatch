#!/bin/bash -l
#SBATCH --job-name=h2_wfa_cuda
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:q6000:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h2cuda_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h2cuda_%j.err

# H2 TEST, NVIDIA side: same parity harness, compiled with nvcc instead of hipcc.
#
# FUSION CRITERION: >=95% agreement (~1000 pairs). The AMD side already scores
# 100% (job 29184155), so this run tests the PORTABILITY claim, not the algebra.
#
# HOW THE NVIDIA PATH WORKS (verified 2026-09-11, no hipify needed):
#   ROCm's hip/hip_runtime.h dispatches on __HIP_PLATFORM_NVIDIA__ to
#   hip/nvidia_detail/nvidia_hip_runtime.h, which wraps <cuda_runtime.h>.
#   So the SAME .hip source compiles under nvcc with that define + includes.
#   No shim install, no hipify step, no new dependency.
#
# CUDA VERSION IS LOAD-BEARING: use 12.4, NOT 13.0.
#   ROCm 6.4.3's nvidia_detail layer targets CUDA 12.x. Under 13.0 it fails on
#   cudaMemLocation / cudaDeviceProp.clockRate / .computeMode.
#
# NOTE: no container is needed here -- nvcc comes from the site's nvhpc module,
# which is why this job sources Lmod rather than using apptainer --nv.

set -o pipefail

R=/kuhpc/sw/rocm/6.4.3
C=/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4
ROOT=/beegfs/a474r867/genoaligner/repo
N_CASES=${N_CASES:-1000}

echo "=== host: $(hostname) ==="
echo "=== H2-CUDA: WFA score parity on NVIDIA ==="
echo

nvidia-smi -L 2>/dev/null || echo "no nvidia-smi"

echo "--- nvcc ---"
"$C/bin/nvcc" --version | tail -2

cd "$ROOT" || exit 1

# sm_75 = Turing (q6000). Add sm_80 for a100 if the node type changes.
echo
echo "--- build harness (nvcc, same source as the AMD path) ---"
"$C/bin/nvcc" -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
    -gencode arch=compute_75,code=sm_75 \
    -I "$R/include" \
    -I "$C/include" \
    -I "$C/targets/x86_64-linux/include" \
    -I . \
    -x cu -o /tmp/wfa_parity_cuda \
    tests/parity/wfa_parity.cpp
RC_BUILD=$?
echo "BUILD_RC=$RC_BUILD"
if [ "$RC_BUILD" -ne 0 ]; then
    echo "H2-CUDA: BUILD FAILED"
    exit 1
fi

echo
echo "--- embedded GPU code ---"
"$C/bin/cuobjdump" --list-elf /tmp/wfa_parity_cuda 2>/dev/null | head -4

echo
echo "--- run ---"
/tmp/wfa_parity_cuda "$N_CASES"
RC=$?

echo
echo "=== H2-CUDA RESULT ==="
if [ "$RC" -eq 0 ]; then echo "H2-CUDA: PASS"; else echo "H2-CUDA: FAIL (rc=$RC)"; fi
exit "$RC"
