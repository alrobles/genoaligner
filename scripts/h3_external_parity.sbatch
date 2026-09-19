#!/bin/bash -l
#SBATCH --job-name=h3_ext_parity
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h3ext_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h3ext_%j.err

# FASE 3: parity of the GPU against EXTERNAL independent implementations.
#
# Distinction from H2: H2 compared the GPU against src/reference, our own DP.
# That is necessary but not sufficient for the Fase 3 claim, because both sides
# would share any conceptual error we made. Here the GPU is compared against
# three implementations we did not write:
#
#     edlib       Myers bit-vector  (Python binding, in ~/.local)
#     rapidfuzz   bit-parallel Levenshtein (Python binding, in ~/.local)
#     SeqAn3      header-only C++23, run as a prebuilt binary
#
# Fusion criterion (masterplan Fase 3): 100% of scores identical. Divergences
# must be understood before proceeding -- a bug, or a documented limit.
#
# Run inside the container so the GPU build matches the validated path, but the
# oracle step needs host python (edlib/rapidfuzz live in ~/.local) -- so the
# harness emits scores, and the comparison runs on the host afterwards.

set -o pipefail

SIF=/beegfs/a474r867/genoaligner/images/genoaligner-compile.sif
ROOT=/beegfs/a474r867/genoaligner/repo
W=/beegfs/a474r867/genoaligner/seqan_probe
N_CASES=${N_CASES:-1000}
EMIT=/beegfs/a474r867/genoaligner/logs/h3emit_${SLURM_JOB_ID:-local}.tsv

echo "=== host: $(hostname) ==="
echo "=== Fase 3: GPU vs external oracles (edlib / rapidfuzz / SeqAn3) ==="

cd "$ROOT" || exit 1

echo "--- build + run harness with --emit (inside container) ---"
apptainer exec --rocm \
    --bind /beegfs:/beegfs \
    --bind "$ROOT":"$ROOT" \
    --pwd "$ROOT" \
    "$SIF" \
    bash -c "
        set -e
        hipcc --version | head -1
        hipcc -O2 -std=c++17 -I. -o /tmp/wfa_parity tests/parity/wfa_parity.cpp
        /tmp/wfa_parity $N_CASES --emit $EMIT
    "
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "HARNESS FAILED (rc=$RC)"
    exit "$RC"
fi

echo
echo "--- build the SeqAn3 oracle if absent (header-only, needs gcc 14.2) ---"
# The cluster default g++ is 11.5, and SeqAn3 3.4 requires GCC >= 12 plus C++23.
# The compiler/gcc/14.2 module does NOT put g++ on PATH, so we use the absolute
# path. The clone is kept under $W/seqan3 (shallow, 4.3 MB).
GXX=/kuhpc/sw/gcc/14.2/bin/g++
if [ ! -x "$W/seqan3_oracle" ]; then
    if [ ! -d "$W/seqan3/include" ]; then
        ( cd "$W" && git clone -q --depth 1 --branch 3.4.0 \
              https://github.com/seqan/seqan3.git ) || echo "clone failed"
    fi
    if [ -d "$W/seqan3/include" ]; then
        "$GXX" -std=c++23 -w -O2 -I "$W/seqan3/include" \
               -o "$W/seqan3_oracle" "$ROOT/tests/parity/seqan3_oracle.cpp" \
            && echo "built seqan3_oracle" \
            || echo "seqan3 oracle build FAILED"
    fi
fi

echo
echo "--- add the SeqAn3 oracle column (host, needs gcc 14.2 libstdc++) ---"
if [ ! -x "$W/seqan3_oracle" ]; then
    echo "WARNING: seqan3_oracle unavailable; comparing with edlib+rapidfuzz only"
    AUG="$EMIT"
else
    LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64 "$W/seqan3_oracle" "$EMIT" "${EMIT%.tsv}.seqan3.tsv"
    AUG="${EMIT%.tsv}.seqan3.tsv"
fi

echo
echo "--- compare against external oracles (host python) ---"
python3 "$ROOT/tests/parity/oracle_external.py" --compare "$AUG"
RC=$?

echo
echo "=== FASE 3 RESULT ==="
if [ "$RC" -eq 0 ]; then echo "FASE 3: PASS"; else echo "FASE 3: FAIL (rc=$RC)"; fi
echo "emitted: $AUG"
exit "$RC"
