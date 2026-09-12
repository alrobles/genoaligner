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

    # traceback_cpu is included: Fase 4's CIGAR reconstruction must clear the
    # same gate as the score kernel before anything touches the GPU.
    for test_src in r1_check wfa_diag traceback_cpu; do
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
        # The traceback binary writes its CIGARs for the external check below.
        local run_args=""
        if [ "$test_src" = "traceback_cpu" ]; then
            run_args="--emit $BUILD_DIR/cigars.tsv"
        fi
        # shellcheck disable=SC2086
        if ! "$bin" $run_args; then
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
#
# Probe by COMPILING AND LINKING a real program, not by preprocessing: a host can
# have the sanitizer headers but lack the runtime library (GCC 11.5 on the KU
# login node has no libasan.so.6.0.0), and a preprocessing probe does not detect
# that. A missing optional tool degrades the gate to pass-1-only and says so; it
# must NOT be reported as a parity/memory failure.
SAN_PROBE="$BUILD_DIR/.san_probe"
SAN_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -O1"
:
if ! printf 'int main(){int a[2]={0,1};return a[0];}\n' \
        | "$CXX" $SAN_FLAGS -x c++ - -o "$SAN_PROBE" >/dev/null 2>&1; then
    echo "!!! sanitizers UNAVAILABLE with $CXX (compile+link probe failed)."
    echo "!!! Memory pass SKIPPED — out-of-bounds reads may reach the GPU undetected."
    echo "!!! Install libasan (e.g. gcc-toolset / libasan package) to enable it."
else
    if ! run_pass "asan+ubsan -O1" "$SAN_FLAGS"; then
        echo "=== CPU GATE FAILED (memory) — do not submit to the cluster ==="
        exit 1
    fi
fi

echo "=== CPU GATE PASSED — algebra + memory indexing verified, cleared for MI210 ==="

# --- stage 3: external CIGAR comparison (Fase 4) -----------------------------
# The traceback's own checks (re-score, well-formedness) can both pass while the
# CIGAR is wrong; three development versions did exactly that. This compares the
# alignment score against edlib, an implementation we did not write.
#
# Optional dependency: if python3 or edlib is missing, say so loudly and do not
# fail the gate -- the C++ checks above already passed, and a missing tool must
# not be reported as a defect (the same rule as the sanitizer probe).
echo
echo "--- Fase 4: external CIGAR comparison (edlib) ---"
CIGARS="$BUILD_DIR/cigars.tsv"

if ! command -v python3 >/dev/null 2>&1; then
    echo "!!! python3 unavailable — external CIGAR check SKIPPED."
    echo "!!! The CIGARs have NOT been compared against an independent implementation."
elif [ ! -s "$CIGARS" ]; then
    echo "!!! no CIGAR emit file at $CIGARS — external check SKIPPED."
elif ! python3 -c "import edlib" >/dev/null 2>&1; then
    echo "!!! edlib unavailable — external CIGAR check SKIPPED."
    echo "!!! pip install -r tests/parity/requirements-oracle.txt"
    echo "!!! The CIGARs have NOT been compared against an independent implementation."
else
    if ! python3 "$REPO_ROOT/tests/parity/check_cigar.py" "$CIGARS"; then
        echo
        echo "=== CPU GATE FAILED (external CIGAR) — do not submit to the cluster ==="
        exit 1
    fi
fi

echo
echo "=== CPU GATE COMPLETE — score, memory and traceback verified ==="
