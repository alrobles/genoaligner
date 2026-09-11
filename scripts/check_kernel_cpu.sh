#!/usr/bin/env bash
# genoaligner — CPU gate for the WFA kernel (R1 verification).
#
# Compiles the REAL kernel source (include/genoaligner/backend/wfa_kernel.hip)
# against a CPU shim and runs it against the independent O(nm) DP reference.
# No GPU, no ROCm, no cluster needed; runs in seconds.
#
# Run this BEFORE every cluster submission. The rule is: nothing touches the
# MI210 until its CPU twin is 100%.
#
# Usage:  scripts/check_kernel_cpu.sh
# Exit:   0 = 100% parity, non-zero = do NOT submit to the cluster.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHIM_DIR="$REPO_ROOT/tests/parity/hip_cpu_shim"
BUILD_DIR="${BUILD_DIR:-/tmp/genoaligner_cpu_check}"
CXX="${CXX:-g++}"

mkdir -p "$BUILD_DIR"

echo "=== genoaligner CPU kernel gate ==="
echo "  repo : $REPO_ROOT"
echo "  cxx  : $($CXX --version | head -1)"
echo

fail=0

for test_src in r1_check wfa_diag; do
    src="$REPO_ROOT/tests/parity/${test_src}.cpp"
    bin="$BUILD_DIR/${test_src}"
    echo "--- build ${test_src} ---"
    if ! "$CXX" -O2 -std=c++17 \
            -I"$SHIM_DIR" -I"$REPO_ROOT" \
            -o "$bin" "$src"; then
        echo "BUILD FAILED: $test_src"
        fail=1
        continue
    fi
    echo "--- run ${test_src} ---"
    if ! "$bin"; then
        fail=1
    fi
    echo
done

if [ "$fail" -ne 0 ]; then
    echo "=== CPU GATE FAILED — do not submit to the cluster ==="
    exit 1
fi

echo "=== CPU GATE PASSED — kernel algebra verified, cleared for MI210 ==="
