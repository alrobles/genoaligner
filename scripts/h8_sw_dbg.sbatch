#!/bin/bash -l
#SBATCH --job-name=h8_swdbg
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h8swd_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h8swd_%j.err

# SW memory fault — LOCATE it, stop guessing.
#
# Two attempts have been spent reasoning about the fault (static/dynamic shared overlap,
# then a missing barrier). Both fixes were defensible and neither stopped it. The next
# step is not a third hypothesis: it is the sanitizer, plus a MINIMAL reproducer that
# shrinks the input until the fault either disappears or is obviously tied to one case.
#
#   (1) run the full test under ROCm's address sanitizer if available;
#   (2) run ONE case at a time, smallest first, printing before and after the launch, so
#       the faulting case is identified by bisection rather than by argument.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
cd "$ROOT" || exit 1
source /kuhpc/sw/lmod/lmod/init/profile 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1

B=/beegfs/a474r867/sw_dbg
echo "=== host: $(hostname) ==="
echo "=== SW fault triage ==="
echo

echo "--- available sanitizer flags ---"
/kuhpc/sw/rocm/6.4.3/bin/hipcc --help 2>&1 | grep -iE "sanitiz" | head -5
echo

echo "--- build with the address sanitizer, if it exists ---"
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O1 -g -std=c++17 -I. -Iinclude \
    -fsanitize=address -o "$B.asan" tests/sw/test_sw_dbg.cpp > /dev/null 2>&1 \
    && echo "asan build OK" || echo "asan build not available"

echo
echo "--- build plain ---"
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -g -std=c++17 -I. -Iinclude \
    -o "$B" tests/sw/test_sw_dbg.cpp || { echo "BUILD FAILED"; exit 1; }
echo "plain build OK"

echo
echo "--- run per-case (bisection by construction) ---"
"$B"
RC=$?

echo
echo "=== SW fault triage DONE (rc=$RC) ==="
exit "$RC"
