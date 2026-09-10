#!/usr/bin/env bash
# Build genoaligner on an AMD (ROCm) node.
# Usage: load the ROCm module first, then run this.
set -euo pipefail

if ! command -v hipcc >/dev/null 2>&1; then
    echo "hipcc not found. Run:  ml load rocm/6.4.3" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build/rocm"

echo "=== ROCm build ==="
echo "hipcc : $(command -v hipcc)"
hipcc --version | head -2

cmake -S "${ROOT}" -B "${BUILD}" \
    -DGENOALIGNER_BACKEND=rocm \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD}" -j "$(nproc)"

echo
echo "=== run probe ==="
"${BUILD}/genoaligner_probe"
