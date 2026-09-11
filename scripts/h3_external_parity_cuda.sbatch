#!/bin/bash -l
#SBATCH --job-name=h3_ext_cuda
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:q6000:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h3extcuda_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h3extcuda_%j.err

# FASE 3, NVIDIA side: GPU vs three external oracles (edlib / rapidfuzz / SeqAn3).
#
# Closes an asymmetry: H2-CUDA (job 29199289) showed 100% CUDA parity against our
# own DP, but the Fase 3 external comparison had only been run on MI210
# (job 29201335). This makes the portability claim hold for the STRONGER
# comparison too.
#
# The kernel source is IDENTICAL to the AMD path. nvcc compiles it directly via
# ROCm's header layer:
#   hip/hip_runtime.h --(when __HIP_PLATFORM_NVIDIA__)--> hip/nvidia_detail/
#     nvidia_hip_runtime.h -> #include <cuda_runtime.h>
# No hipify, no shim install.
#
# CUDA VERSION IS LOAD-BEARING: 12.4, not 13.0. ROCm 6.4.3's nvidia_detail layer
# targets CUDA 12.x and fails under 13.0 on cudaMemLocation / cudaDeviceProp.
#
# GPU arch: q6000 is Quadro RTX 6000 = Turing = sm_75.

set -o pipefail

R=/kuhpc/sw/rocm/6.4.3
C=/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4
ROOT=/beegfs/a474r867/genoaligner/repo
W=/beegfs/a474r867/genoaligner/seqan_probe
N_CASES=${N_CASES:-1000}
EMIT=/beegfs/a474r867/genoaligner/logs/h3emitcuda_${SLURM_JOB_ID:-local}.tsv

echo "=== host: $(hostname) ==="
echo "=== Fase 3 CUDA: GPU vs external oracles ==="
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi"

echo "--- nvcc ---"
"$C/bin/nvcc" --version | tail -2

cd "$ROOT" || exit 1

echo
echo "--- build harness with --emit (nvcc, same source as AMD) ---"
"$C/bin/nvcc" -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
    -gencode arch=compute_75,code=sm_75 \
    -I "$R/include" \
    -I "$C/include" \
    -I "$C/targets/x86_64-linux/include" \
    -I . \
    -x cu -o /tmp/wfa_parity_cuda \
    tests/parity/wfa_parity.cpp
RC=$?
echo "BUILD_RC=$RC"
if [ "$RC" -ne 0 ]; then
    echo "FASE 3 CUDA: BUILD FAILED"
    exit 1
fi

"$C/bin/cuobjdump" --list-elf /tmp/wfa_parity_cuda 2>/dev/null | head -3

echo
echo "--- run ---"
/tmp/wfa_parity_cuda "$N_CASES" --emit "$EMIT"
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "HARNESS FAILED (rc=$RC)"
    exit "$RC"
fi

echo
echo "--- build the SeqAn3 oracle if absent (header-only, needs gcc 14.2) ---"
GXX=/kuhpc/sw/gcc/14.2/bin/g++
if [ ! -x "$W/seqan3_oracle" ]; then
    if [ ! -d "$W/seqan3/include" ]; then
        ( cd "$W" && git clone -q --depth 1 --branch 3.4.0 \
              https://github.com/seqan/seqan3.git ) || echo "clone failed"
    fi
    if [ -d "$W/seqan3/include" ]; then
        "$GXX" -std=c++23 -w -O2 -I "$W/seqan3/include" \
               -o "$W/seqan3_oracle" "$ROOT/tests/parity/seqan3_oracle.cpp" \
            && echo "built seqan3_oracle" || echo "seqan3 oracle build FAILED"
    fi
fi

echo
echo "--- add the SeqAn3 oracle column ---"
if [ ! -x "$W/seqan3_oracle" ]; then
    echo "WARNING: seqan3_oracle unavailable; comparing with edlib+rapidfuzz only"
    AUG="$EMIT"
else
    LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64 "$W/seqan3_oracle" "$EMIT" "${EMIT%.tsv}.seqan3.tsv"
    AUG="${EMIT%.tsv}.seqan3.tsv"
fi

echo
echo "--- compare against external oracles ---"
python3 "$ROOT/tests/parity/oracle_external.py" --compare "$AUG"
RC=$?

echo
echo "=== FASE 3 CUDA RESULT ==="
if [ "$RC" -eq 0 ]; then echo "FASE 3 CUDA: PASS"; else echo "FASE 3 CUDA: FAIL (rc=$RC)"; fi
echo "emitted: $AUG"
exit "$RC"
