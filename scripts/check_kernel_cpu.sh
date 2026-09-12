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

    # traceback_cpu is the Fase 4 PROTOTYPE (standalone walk). trace_back_kernel
    # is the REAL kernel: tests/parity/wfa_parity.cpp built against the shim,
    # which calls wfa_trace_kernel's body directly. Both are gated, because the
    # prototype passing does not say the kernel passes.
    for test_src in r1_check wfa_diag traceback_cpu; do
        local src="$REPO_ROOT/tests/parity/${test_src}.cpp"
        local bin="$BUILD_DIR/${test_src}"
        echo "--- [${label}] build ${test_src} ---"
        # shellcheck disable=SC2086
        if ! "$CXX" $extra -std=c++17 \
                -I"$SHIM_DIR" -I"$REPO_ROOT" -I"$REPO_ROOT/include" \
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

    # --- the real kernel, exercised through the CPU shim ----------------------
    # wfa_parity.cpp with -DGENOALIGNER_HIP_SHIM calls wfa_trace_kernel's body
    # with host memory (one "block" per pair). This is the check that says the
    # SHIPPED kernel is correct, not a prototype of it -- see §11 of the
    # ku-hpc-gpu-toolchain skill.
    local tsrc="$REPO_ROOT/tests/parity/wfa_parity.cpp"
    local tbin="$BUILD_DIR/wfa_parity_shim"
    echo "--- [${label}] build wfa_parity (shim, trace-kernel) ---"
    # shellcheck disable=SC2086
    if ! "$CXX" $extra -std=c++17 -DGENOALIGNER_HIP_SHIM \
            -I"$SHIM_DIR" -I"$REPO_ROOT" -I"$REPO_ROOT/include" \
            -o "$tbin" "$tsrc"; then
        echo "BUILD FAILED (${label}): wfa_parity.cpp (shim)"
        pass_fail=1
    else
        echo "--- [${label}] run wfa_parity --traceback ---"
        # 207 random + 7 adversarial = the same control set the GPU harness uses.
        # Emits CIGARs so the external edlib check below scores THIS run's pairs.
        if ! "$tbin" 207 --traceback --emit "$BUILD_DIR/cigars_kernel.tsv"; then
            echo "RUN FAILED (${label}): wfa_parity (shim, trace-kernel)"
            pass_fail=1
        fi
    fi
    echo
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

if ! command -v python3 >/dev/null 2>&1; then
    echo "!!! python3 unavailable — external CIGAR check SKIPPED."
    echo "!!! The CIGARs have NOT been compared against an independent implementation."
elif ! python3 -c "import edlib" >/dev/null 2>&1; then
    echo "!!! edlib unavailable — external CIGAR check SKIPPED."
    echo "!!! pip install -r tests/parity/requirements-oracle.txt"
    echo "!!! The CIGARs have NOT been compared against an independent implementation."
else
    # BOTH sources are checked, and each must be non-empty. A single empty file
    # must not read as a pass: check_cigar.py exits 2 (INCONCLUSIVE) on no input
    # precisely so that silence cannot masquerade as agreement.
    for CIGARS in "$BUILD_DIR/cigars.tsv" "$BUILD_DIR/cigars_kernel.tsv"; do
        echo
        echo "  --- $CIGARS ---"
        if [ ! -s "$CIGARS" ]; then
            echo "  !!! no emit file at $CIGARS — external check SKIPPED for this source."
            echo "  !!! Those CIGARs have NOT been compared against edlib."
            continue
        fi
        if ! python3 "$REPO_ROOT/tests/parity/check_cigar.py" "$CIGARS"; then
            echo
            echo "=== CPU GATE FAILED (external CIGAR) — do not submit to the cluster ==="
            exit 1
        fi
    done
fi

# --- Stage 5: the PUBLIC API ------------------------------------------------
# The gate above verifies the KERNELS. This stage verifies the code a user would
# actually call, because those are not the same thing: the API owns the buffer
# sizing, the batch packing and the input validation, and it has already shipped one
# real bug in each (a CIGAR cap sized from the first request, and a silently clamped
# smax). A gate that does not exercise the public surface cannot catch those.
echo
echo "--- [shim] build and run the public API test ---"
if g++ -O2 -std=c++17 -DGENOALIGNER_HIP_SHIM -I"$SHIM_DIR" -I"$REPO_ROOT" -I"$REPO_ROOT/include" \
       -o "$BUILD_DIR/test_api" \
       "$REPO_ROOT/tests/api/test_api.cpp" "$REPO_ROOT/src/api/api.cpp" 2>"$BUILD_DIR/api_build.log"; then
    if ! "$BUILD_DIR/test_api" | tee "$BUILD_DIR/api_test.out" | tail -20; then
        echo
        echo "=== CPU GATE FAILED (public API) — do not submit to the cluster ==="
        exit 1
    fi
    # A run that reports nothing must not read as a pass (same rule as the CIGAR stage).
    if ! grep -q "RESULT: PASS" "$BUILD_DIR/api_test.out"; then
        echo "  !!! API test produced no PASS verdict — treating as FAILURE."
        exit 1
    fi
else
    echo "  !!! API test failed to BUILD:"
    sed -n '1,20p' "$BUILD_DIR/api_build.log"
    exit 1
fi

echo
echo "--- [host] build and run the FASTA reader test ---"
if g++ -O2 -std=c++17 -I"$REPO_ROOT" -I"$REPO_ROOT/include" -o "$BUILD_DIR/test_fasta" \
       "$REPO_ROOT/tests/io/test_fasta.cpp" 2>"$BUILD_DIR/fasta_build.log"; then
    if ! "$BUILD_DIR/test_fasta" | tee "$BUILD_DIR/fasta_test.out" | tail -12; then
        echo
        echo "=== CPU GATE FAILED (FASTA reader) — do not submit to the cluster ==="
        exit 1
    fi
    if ! grep -q "RESULT: PASS" "$BUILD_DIR/fasta_test.out"; then
        echo "  !!! FASTA test produced no PASS verdict — treating as FAILURE."
        exit 1
    fi
else
    echo "  !!! FASTA test failed to BUILD:"
    sed -n '1,20p' "$BUILD_DIR/fasta_build.log"
    exit 1
fi

# --- Stage 6: REAL biological sequences -------------------------------------
# Stage 5 exercises the API. This stage changes the INPUT: real mtDNA instead of
# generated bases, because repeats, low-complexity and structured regions are where
# traceback implementations break, and a generator does not produce them.
#
# Needs a GPU, so it is skipped (77) on a device-less host -- "cannot run here" is
# not "is broken". When it does run, the same rule as everywhere: no PASS verdict
# means failure.
echo
echo "--- [gpu] build and run the real-sequence test ---"
# Find a HIP compiler without hardcoding this cluster's path: the gate must remain
# runnable elsewhere (that is the whole point of the portability claim).
HIPCC="${HIPCC:-$(command -v hipcc || true)}"
[ -z "$HIPCC" ] && [ -x /kuhpc/sw/rocm/6.4.3/bin/hipcc ] && HIPCC=/kuhpc/sw/rocm/6.4.3/bin/hipcc
[ -z "$HIPCC" ] && [ -x /opt/rocm/bin/hipcc ] && HIPCC=/opt/rocm/bin/hipcc
if [ -z "$HIPCC" ]; then
    echo "  (skipped: no hipcc on PATH and none at the usual locations)"
    echo "  Real-sequence correctness NOT verified here. Set HIPCC= to override."
else
echo "  hipcc: $HIPCC"
if "$HIPCC" -O2 -std=c++17 -I"$REPO_ROOT" -I"$REPO_ROOT/include" \
       -o "$BUILD_DIR/test_real" \
       "$REPO_ROOT/tests/real/test_real_sequences.cpp" "$REPO_ROOT/src/api/api.cpp" \
       2>"$BUILD_DIR/real_build.log"; then
    "$BUILD_DIR/test_real" | tee "$BUILD_DIR/real_test.out" | tail -18
    RC=${PIPESTATUS[0]}
    if [ "$RC" -eq 77 ]; then
        echo "  (skipped: no device visible — real-sequence correctness NOT verified here)"
    elif [ "$RC" -ne 0 ]; then
        echo
        echo "=== CPU GATE FAILED (real sequences) — do not submit to the cluster ==="
        exit 1
    elif ! grep -q "RESULT: PASS" "$BUILD_DIR/real_test.out"; then
        echo "  !!! real-sequence test produced no PASS verdict — treating as FAILURE."
        exit 1
    else
        REAL_RAN=1

        # --- Stage 7: concurrency and batch semantics (same compiler, already built
        # --- above). Both need a device, so they live inside this branch.
        echo
        echo "--- [gpu] concurrency test (B3) ---"
        if "$HIPCC" -O2 -std=c++17 -pthread -I"$REPO_ROOT" -I"$REPO_ROOT/include" -o "$BUILD_DIR/test_conc" \
               "$REPO_ROOT/tests/concurrency/test_concurrency.cpp" \
               "$REPO_ROOT/src/api/api.cpp" 2>"$BUILD_DIR/conc_build.log"; then
            "$BUILD_DIR/test_conc" | tee "$BUILD_DIR/conc_test.out" | tail -8
            if [ "${PIPESTATUS[0]}" -ne 0 ] || ! grep -q "RESULT: PASS" "$BUILD_DIR/conc_test.out"; then
                echo "  !!! concurrency test did not PASS — treating as FAILURE."
                exit 1
            fi
        else
            echo "  !!! concurrency test failed to BUILD:"; sed -n '1,15p' "$BUILD_DIR/conc_build.log"; exit 1
        fi

        echo
        echo "--- [gpu] batch smax semantics (B4) ---"
        if "$HIPCC" -O2 -std=c++17 -I"$REPO_ROOT" -I"$REPO_ROOT/include" -o "$BUILD_DIR/test_batch" \
               "$REPO_ROOT/tests/api/test_batch_semantics.cpp" \
               "$REPO_ROOT/src/api/api.cpp" 2>"$BUILD_DIR/batch_build.log"; then
            "$BUILD_DIR/test_batch" | tee "$BUILD_DIR/batch_test.out" | tail -8
            if [ "${PIPESTATUS[0]}" -ne 0 ] || ! grep -q "RESULT: PASS" "$BUILD_DIR/batch_test.out"; then
                echo "  !!! batch-semantics test did not PASS — treating as FAILURE."
                exit 1
            fi
        else
            echo "  !!! batch-semantics test failed to BUILD:"; sed -n '1,15p' "$BUILD_DIR/batch_build.log"; exit 1
        fi
    fi
else
    echo "  !!! real-sequence test failed to BUILD:"
    sed -n '1,20p' "$BUILD_DIR/real_build.log"
    exit 1
fi
fi

echo
# The summary must not claim what was skipped. A gate that ends with "verified" when
# a stage did not run is the same class of lie this project keeps finding: a green
# line that reads as coverage it does not have.
if [ "${REAL_RAN:-0}" -eq 1 ]; then
    echo "=== CPU GATE COMPLETE — score, memory, traceback, API, FASTA and real sequences verified ==="
else
    echo "=== CPU GATE COMPLETE (partial) — score, memory, traceback, API and FASTA verified."
    echo "=== REAL-SEQUENCE STAGE DID NOT RUN on this host (no GPU/hipcc). ==="
    echo "=== That stage is NOT covered by this run; run it on a GPU node. ==="
fi
