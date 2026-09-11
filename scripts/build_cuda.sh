#!/usr/bin/env bash
# Build genoaligner on an NVIDIA (CUDA) node.
#
# HOW THIS WORKS (and how it does NOT):
#   The compiler is nvcc, NOT hipcc. hipcc is AMD's clang wrapper and never
#   invokes nvcc, so it cannot target NVIDIA. What makes the same .hip source
#   work is ROCm's header layer:
#
#     hip/hip_runtime.h
#       #if defined(__HIP_PLATFORM_NVIDIA__) && !defined(__HIP_PLATFORM_AMD__)
#         #include <hip/nvidia_detail/nvidia_hip_runtime.h> -> <cuda_runtime.h>
#
#   So this script passes -D__HIP_PLATFORM_NVIDIA__ and both ROCm's and CUDA's
#   include dirs. No hipify step and no shim install are needed.
#
# CUDA VERSION: 12.4, not 13.0. ROCm 6.4.3's nvidia_detail layer targets CUDA
# 12.x; under 13.0 it fails on cudaMemLocation and cudaDeviceProp members.
#
# Usage:  scripts/build_cuda.sh            # q6000 (Turing, sm_75)
#         GENOALIGNER_ARCH=80 scripts/build_cuda.sh   # A100 (sm_80)
set -euo pipefail

ROCM_ROOT="${GENOALIGNER_ROCM_ROOT:-/kuhpc/sw/rocm/6.4.3}"
CUDA_ROOT="${GENOALIGNER_CUDA_ROOT:-/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4}"
NVCC="${CUDA_ROOT}/bin/nvcc"
ARCH="${GENOALIGNER_ARCH:-75}"     # 75 = Turing/q6000, 80 = Ampere/a100

[ -x "$NVCC" ] || {
    echo "nvcc not found at $NVCC" >&2
    echo "Set GENOALIGNER_CUDA_ROOT to a CUDA 12.x toolkit (13.0 does NOT work:" >&2
    echo "ROCm 6.4.3's hip/nvidia_detail layer targets CUDA 12.x)." >&2
    exit 1
}
[ -f "${ROCM_ROOT}/include/hip/hip_runtime.h" ] || {
    echo "HIP headers not found under ${ROCM_ROOT}; the NVIDIA path needs them" >&2
    echo "(that is where hip/nvidia_detail lives)." >&2
    exit 1
}

# cmake is not on PATH via the KU modules; add the known locations.
if ! command -v cmake >/dev/null 2>&1; then
    for c in /kuhpc/sw/cmake/3.30.3/gcc/14.2/bin \
             /kuhpc/sw/cmake/3.30.3/gcc/11.4/bin; do
        [ -x "$c/cmake" ] && export PATH="$c:$PATH" && break
    done
fi
command -v cmake >/dev/null 2>&1 || {
    echo "cmake not found (searched module + /kuhpc/sw/cmake/*)" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build/cuda"

echo "=== genoaligner CUDA build ==="
echo "nvcc   : $NVCC"
"$NVCC" --version | tail -2
echo "rocm   : $ROCM_ROOT  (HIP header layer)"
echo "cuda   : $CUDA_ROOT"
echo "arch   : sm_${ARCH}"
echo "cmake  : $(command -v cmake)"

cmake -S "${ROOT}" -B "${BUILD}" \
    -DCMAKE_CXX_COMPILER="${NVCC}" \
    -DGENOALIGNER_BACKEND=cuda \
    -DGENOALIGNER_CUDA_ROOT="${CUDA_ROOT}" \
    -DGENOALIGNER_ROCM_ROOT="${ROCM_ROOT}" \
    -DCMAKE_CXX_FLAGS="-w -gencode arch=compute_${ARCH},code=sm_${ARCH}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD}" -j "$(nproc)"

echo
echo "=== embedded GPU code ==="
"${CUDA_ROOT}/bin/cuobjdump" --list-elf "${BUILD}/genoaligner_probe" 2>/dev/null | head -3

echo
echo "=== run probe ==="
"${BUILD}/genoaligner_probe"
