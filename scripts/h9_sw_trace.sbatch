#!/bin/bash -l
#SBATCH --job-name=h9_sw_trace
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h9sw_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h9sw_%j.err

# Smith-Waterman traceback on GPU (Fase B).
#
# What this job establishes, stated narrowly:
#   1. sw_trace_kernel on a real device: score == sw_score_kernel == reference,
#      CIGAR well-formed on the aligned span, re-scores exactly, and equals the
#      independent matrix walk -- the same checks as the CPU gate.
#   2. EXTERNAL ORACLE: the emitted cases are scored by SeqAn3 (local, affine
#      gaps; mapping verified by seqan3_sw_oracle --selftest). The score must
#      agree; positions may differ on ties.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
LOGD=/beegfs/a474r867/genoaligner/logs
PROBE=/beegfs/a474r867/genoaligner/seqan_probe
cd "$ROOT" || exit 1
source /kuhpc/sw/lmod/lmod/init/profile 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1

B=/beegfs/a474r867/sw_trace_gpu
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. -Iinclude \
    -o "$B" tests/sw/test_sw_trace_gpu.cpp || { echo "BUILD FAILED"; exit 1; }

echo "=== host: $(hostname) ==="
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
/kuhpc/sw/rocm/6.4.3/bin/rocm-smi --showpids 2>/dev/null || true
echo "=== SW traceback on GPU ==="
echo
TSV="$LOGD/sw_trace_${SLURM_JOB_ID}.tsv"
"$B" --emit "$TSV" | tee "$LOGD/sw_trace_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
[ "$RC" -ne 0 ] && { echo "=== H9-SW2: FAIL (rc=$RC) ==="; exit "$RC"; }

echo
echo "=== external oracle: SeqAn3 (local, affine) ==="
# SeqAn3 needs g++14.2 headers+libstdc++; the oracle binary is prebuilt in
# $PROBE (source: tests/sw/seqan3_sw_oracle.cpp, kept under version control).
LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64 "$PROBE/seqan3_sw_oracle" --selftest || {
    echo "=== H9-SW2: FAIL (oracle selftest) ==="; exit 1; }
LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64 \
    "$PROBE/seqan3_sw_oracle" "$TSV" "$LOGD/sw_trace_oracle_${SLURM_JOB_ID}.tsv" || {
    echo "=== H9-SW2: FAIL (seqan3 disagrees) ==="; exit 1; }

echo
echo "=== H9-SW2 RESULT ==="
echo "H9-SW2: PASS"
exit 0
