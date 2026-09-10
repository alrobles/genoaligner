#!/usr/bin/env bash
# Build genoaligner on an NVIDIA (CUDA) node through the HIP shim.
# On CUDA nodes hipcc is provided by the NVIDIA HIP toolchain or the ROCm
# package's nvcc wrapper; the same source must compile unchanged.
set -euo pipefail

if ! command -v hipcc >/dev/null 2>&1; then
    echo "hipcc not found. Run:  ml load cuda/13.0   (or the HIP shim)" >&2
    exit 1
fi

# cmake is not exposed on PATH by the KU HPC modules; add the known location.
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

echo "=== CUDA build (via HIP) ==="
echo "hipcc : $(command -v hipcc)"
hipcc --version | head -2
echo "cmake : $(command -v cmake)"

cmake -S "${ROOT}" -B "${BUILD}" \
    -DCMAKE_CXX_COMPILER="$(command -v hipcc)" \
    -DGENOALIGNER_BACKEND=cuda \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD}" -j "$(nproc)"

echo
echo "=== run probe ==="
"${BUILD}/genoaligner_probe"
