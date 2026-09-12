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
