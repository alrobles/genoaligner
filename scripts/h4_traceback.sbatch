#!/bin/bash -l
#SBATCH --job-name=h4_traceback
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h4_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h4_%j.err

# H4 TEST: Fase 4 traceback (CIGAR reconstruction) on the MI210.
#
# The CPU gate (scripts/check_kernel_cpu.sh) has already run the SAME kernel body
# under the shim, so this job is about the things the shim CANNOT prove:
#   - that the kernel compiles under hipcc at all
#   - real barriers, warp behaviour, block-size limits
#   - the launch itself and dynamic shared memory
#   - the device-side memory traffic (workspace read-back)
#
# What it is NOT: a first look at the algorithm. If this fails on algebra, the
# gate had a hole.
#
# Builds with hipcc directly (no container) so the compile path is the same one
# the R1/H2 jobs use.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
N_CASES=${N_CASES:-207}
CHUNK_BYTES=${CHUNK_BYTES:-49152}
# Emit to beegfs, NOT /tmp. A /tmp path dies with the job and left the Fase 4
# gate unable to run its external stage on the GPU output (the CIGARs had to be
# re-emitted). The node-local path was fine for the run and useless afterwards.
OUT_DIR=/beegfs/a474r867/genoaligner/out
mkdir -p "$OUT_DIR"
CIGAR_TSV=$OUT_DIR/cigars_gpu_mi210_${SLURM_JOB_ID}.tsv

echo "=== host: $(hostname) ==="
echo "=== H4: Fase 4 traceback on MI210 ==="
echo

source /kuhpc/sw/lmod/9.3/init/bash 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1

which hipcc || { echo "hipcc not found"; exit 1; }
hipcc --version | head -1

cd "$ROOT" || exit 1

echo
echo "--- build harness (hipcc, real device code) ---"
hipcc -O2 -std=c++17 -I. \
      -o /tmp/wfa_parity_trace \
      tests/parity/wfa_parity.cpp
RC_BUILD=$?
echo "BUILD_RC=$RC_BUILD"
if [ "$RC_BUILD" -ne 0 ]; then
    echo "H4: BUILD FAILED (the shim could not have caught this)"
    exit 1
fi

echo
echo "--- embedded GPU code ---"
(llvm-objdump --list-symbols /tmp/wfa_parity_trace 2>/dev/null \
    || /opt/rocm/llvm/bin/llvm-objdump --list-symbols /tmp/wfa_parity_trace 2>/dev/null) \
    | grep -i "wfa_trace_kernel" | head -3

echo
echo "--- run: traceback, default chunk ($CHUNK_BYTES B) ---"
/tmp/wfa_parity_trace "$N_CASES" --traceback --chunk-bytes "$CHUNK_BYTES" \
    --emit "$CIGAR_TSV"
RC=$?

echo
echo "--- run: traceback with NO shared memory (global fallback) ---"
# This is the measurement the design decision rests on: how much slower is the
# zero-shared-memory path, i.e. what does the shared buffer actually buy?
/tmp/wfa_parity_trace "$N_CASES" --traceback --chunk-bytes 0

echo
echo "--- stage 3: CIGAR score vs edlib (external implementation) on GPU output ---"
# The third stage of the gate, and the only one that is genuinely external. The
# in-harness checks (re-score, well-formedness) are self-consistency: both can
# pass while the CIGAR is wrong, and that actually happened during development.
# Until this stage runs on the GPU-emitted CIGARs, the gate's external arm had
# only ever been exercised on CPU output.
python3 tests/parity/check_cigar.py "$CIGAR_TSV"
RC_EDLIB=$?
if [ "$RC_EDLIB" -ne 0 ]; then RC=1; fi

echo
echo "=== H4 RESULT ==="
if [ "$RC" -eq 0 ]; then echo "H4: PASS"; else echo "H4: FAIL (rc=$RC)"; fi
exit "$RC"
