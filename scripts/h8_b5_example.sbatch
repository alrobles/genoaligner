#!/bin/bash -l
#SBATCH --job-name=h8_b5
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h8b5_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h8b5_%j.err

# Production flag B5 — the README example must be REAL output, not invented.
#
# Two things are proven here, both of which the README then claims:
#   1. the example builds against the INSTALLED tree, using the include form a
#      consumer writes (<genoaligner/api.hpp> with -I<prefix>/include), not the
#      in-tree relative path the internal tests use;
#   2. the output pasted into the README is what the program actually prints.
set -o pipefail

GAINST=/beegfs/a474r867/gainst
cd /beegfs/a474r867/genoaligner/repo || exit 1

echo "=== host: $(hostname) ==="
echo "=== H8-B5: README example, built against the install ==="
echo

# REINSTALL FIRST. The first version of this job compiled against whatever was
# already in $GAINST, so it tested a STALE library and reported the previous
# revision's bug -- a full debugging cycle spent on an artifact that did not
# contain the fix. A test whose subject is "the installed tree" must build the
# installed tree, every run.
echo "--- rebuild and reinstall (the subject of this test) ---"
source /kuhpc/sw/lmod/lmod/init/profile 2>/dev/null
ml purge >/dev/null 2>&1
ml load compiler/gcc/14.2 >/dev/null 2>&1
ml load cmake/3.30.3 >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1
BD=/beegfs/a474r867/binst_h8b5
LOGD=/beegfs/a474r867/genoaligner/logs
mkdir -p "$LOGD"
rm -rf "$BD" "$GAINST"
cmake -S . -B "$BD" -DCMAKE_CXX_COMPILER="$(command -v hipcc)" \
      -DGENOALIGNER_BACKEND=rocm -DCMAKE_INSTALL_PREFIX="$GAINST" > "$LOGD/b5_cm.log" 2>&1
echo "cmake rc=$?"
cmake --build "$BD" -j8 > "$LOGD/b5_bd.log" 2>&1
RC_BD=$?
echo "build rc=$RC_BD"
if [ "$RC_BD" -ne 0 ]; then
    echo "--- build errors (tail) ---"
    grep -iE "error|Error" "$LOGD/b5_bd.log" | head -15
    echo "H8-B5: BUILD OF THE LIBRARY FAILED"; exit 1
fi
cmake --install "$BD" > "$LOGD/b5_in.log" 2>&1
echo "install rc=$?"
echo "HEAD: $(git rev-parse --short HEAD)"

echo
echo "--- installed tree ---"
find "$GAINST" -type f | sort

echo
echo "--- build the example as a CONSUMER would ---"
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 \
    -I"$GAINST/include" -o /tmp/consumer examples/align_fasta.cpp \
    -L"$GAINST/lib" -lgenoaligner
RC=$?
echo "BUILD_RC=$RC"
[ "$RC" -ne 0 ] && { echo "H8-B5: BUILD FAILED"; exit 1; }

echo
echo "--- run it (this output is what the README pastes) ---"
/tmp/consumer tests/data/mtdna_human.fa
RC=$?

echo
echo "=== H8-B5 RESULT ==="
if [ "$RC" -eq 0 ]; then echo "H8-B5: PASS"; else echo "H8-B5: FAIL (rc=$RC)"; fi
exit "$RC"
