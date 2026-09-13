#!/bin/bash -l
#SBATCH --job-name=h11_sw_cuda
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h11cuda_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h11cuda_%j.err

# H11-CUDA: the SW path on NVIDIA, same sources as the MI210 jobs.
#
# What this run establishes, stated narrowly:
#   1. sw_score_kernel (warp-per-row scan) on a REAL warpSize=32 device --
#      every prior run was warpSize=64 (MI210) or emulated (shim at 32/64).
#   2. sw_trace_kernel on the same device: score == reference, CIGAR
#      well-formed, re-scores, equals the independent walk.
#   3. The PUBLIC API (align_sw_batch) on the device, incl. --emit-sw for the
#      SeqAn3 oracle.
#
# Mechanism (see scripts/build_cuda.sh): nvcc compiles the .hip sources through
# ROCm's hip/nvidia_detail header layer + CUDA 12.4. -lcuda is needed because
# the trace/API paths query device attributes (driver API via nvidia_detail).
set -o pipefail

R=/kuhpc/sw/rocm/6.4.3
C=/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4
ROOT=/beegfs/a474r867/genoaligner/repo
LOGD=/beegfs/a474r867/genoaligner/logs
PROBE=/beegfs/a474r867/genoaligner/seqan_probe
B=/beegfs/a474r867/genoaligner/gate_build_h11
mkdir -p "$B" "$LOGD"
cd "$ROOT" || exit 1

echo "=== host: $(hostname) ==="
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
"$C/bin/nvcc" --version | tail -2
echo

NVCC_FLAGS="-w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
    -gencode arch=compute_70,code=sm_70 \
    -I $R/include -I $C/include -I $C/targets/x86_64-linux/include \
    -I . -I include"

build() {
    local out=$1; shift
    echo "--- nvcc -> $out"
    "$C/bin/nvcc" $NVCC_FLAGS -lcuda -L "$C/lib64/stubs" -x cu -o "$out" "$@" \
        2>"$B/$(basename "$out").build.log" || {
        echo "BUILD FAILED: $out"; sed -n '1,25p' "$B/$(basename "$out").build.log"
        exit 1
    }
}

build "$B/sw_gpu"       tests/sw/test_sw_gpu.cpp
build "$B/sw_trace_gpu" tests/sw/test_sw_trace_gpu.cpp
build "$B/test_api"     tests/api/test_api.cpp src/api/api.cpp

echo
echo "=== embedded GPU code ==="
"$C/bin/cuobjdump" --list-elf "$B/sw_gpu" 2>/dev/null | head -4

RC=0

echo
echo "=== [1/3] sw_score_kernel on device (test_sw_gpu, correctness only) ==="
"$B/sw_gpu" || RC=1

echo
echo "=== [2/3] sw_trace_kernel on device ==="
TSV="$LOGD/sw_trace_cuda_${SLURM_JOB_ID}.tsv"
"$B/sw_trace_gpu" --emit "$TSV" | tee "$LOGD/sw_trace_cuda_${SLURM_JOB_ID}.txt" || RC=1

echo
echo "=== [3/3] public API on device (incl. SW section + emit) ==="
"$B/test_api" --emit-sw "$LOGD/api_sw_cuda_${SLURM_JOB_ID}.tsv" | tail -40 || RC=1

echo
echo "=== external oracle: SeqAn3 on both emitted TSVs ==="
LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64 "$PROBE/seqan3_sw_oracle" --selftest || RC=1
LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64 \
    "$PROBE/seqan3_sw_oracle" "$TSV" "$LOGD/sw_trace_oracle_cuda_${SLURM_JOB_ID}.tsv" || RC=1
LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64 \
    "$PROBE/seqan3_sw_oracle" "$LOGD/api_sw_cuda_${SLURM_JOB_ID}.tsv" \
    "$LOGD/api_sw_oracle_cuda_${SLURM_JOB_ID}.tsv" || RC=1

echo
echo "=== H11-CUDA RESULT ==="
if [ "$RC" -eq 0 ]; then echo "H11-CUDA: PASS"; else echo "H11-CUDA: FAIL"; fi
exit "$RC"
