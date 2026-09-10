#!/usr/bin/env bash
# Build genoaligner on an AMD (ROCm) node.
# Usage: load the ROCm module first, then run this.
set -euo pipefail

if ! command -v hipcc >/dev/null 2>&1; then
    echo "hipcc not found. Run:  ml load rocm/6.4.3" >&2
    exit 1
fi

# cmake on KU HPC lives under /kuhpc/sw/cmake/<ver>/<compiler>/bin and is NOT
# added to PATH by any module (the 'cmake' modules silently do nothing). Add
# the known location as a fallback.
if ! command -v cmake >/dev/null 2>&1; then
    for c in /kuhpc/sw/cmake/3.30.3/gcc/14.2/bin \
             /kuhpc/sw/cmake/3.30.3/gcc/11.4/bin; do
        [ -x "$c/cmake" ] && export PATH="$c:$PATH" && break
    done
fi
command -v cmake >/dev/null 2>&1 || {
    echo "cmake not found (searched module + /kuhpc/sw/cmake/*)" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build/rocm"

echo "=== ROCm build ==="
echo "hipcc : $(command -v hipcc)"
hipcc --version | head -2
echo "cmake : $(command -v cmake)"

cmake -S "${ROOT}" -B "${BUILD}" \
    -DCMAKE_CXX_COMPILER="$(command -v hipcc)" \
    -DGENOALIGNER_BACKEND=rocm \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD}" -j "$(nproc)"

echo
echo "=== run probe ==="
"${BUILD}/genoaligner_probe"
