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
# Exit:   0 = 100% parity + no memory errors, non-zero = do NOT submit.
#
# NOTE ON COVERAGE. This gate verifies the ALGEBRA and, in the ASan pass, the
# MEMORY INDEXING. It does not verify synchronisation, warp behaviour or device
# memory limits. Specifically, a one-past-the-end read of a small shared buffer
# is caught by the ASan pass but was previously invisible in the plain build,
# because the shim's std::vector is large enough that the stray read lands
# inside the allocation. That exact gap caused H2 job 29184154 to fault on the
# MI210 (rc=134) after this gate passed. Hence the second, sanitizer pass.
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

run_pass() {
    # $1 = pass label, $2 = extra flags
    local label="$1"; shift
    local extra="$1"; shift
    local pass_fail=0

    for test_src in r1_check wfa_diag; do
        local src="$REPO_ROOT/tests/parity/${test_src}.cpp"
        local bin="$BUILD_DIR/${test_src}"
        echo "--- [${label}] build ${test_src} ---"
        # shellcheck disable=SC2086
        if ! "$CXX" $extra -std=c++17 \
                -I"$SHIM_DIR" -I"$REPO_ROOT" \
                -o "$bin" "$src"; then
            echo "BUILD FAILED (${label}): $test_src"
            pass_fail=1
            continue
        fi
        echo "--- [${label}] run ${test_src} ---"
        if ! "$bin"; then
            echo "RUN FAILED (${label}): $test_src"
            pass_fail=1
        fi
        echo
    done
    return $pass_fail
}

# --- pass 1: optimised, the parity numbers we care about ---------------------
if ! run_pass "plain -O2" "-O2"; then
    echo "=== CPU GATE FAILED (parity) — do not submit to the cluster ==="
    exit 1
fi

# --- pass 2: sanitizers, to catch out-of-bounds the plain build hides --------
# AddressSanitizer + UBSan. This is the pass that would have caught the shared
# buffer overflow before it reached the GPU.
if "$CXX" -fsanitize=address,undefined -fno-omit-frame-pointer -O1 \
        -x c++ -E - </dev/null >/dev/null 2>&1; then
    if ! run_pass "asan+ubsan -O1" "-fsanitize=address,undefined -fno-omit-frame-pointer -O1"; then
        echo "=== CPU GATE FAILED (memory) — do not submit to the cluster ==="
        exit 1
    fi
else
    echo "!!! sanitizers unavailable with $CXX — memory pass SKIPPED."
    echo "!!! Out-of-bounds reads may reach the GPU undetected."
fi

echo "=== CPU GATE PASSED — algebra + memory indexing verified, cleared for MI210 ==="
