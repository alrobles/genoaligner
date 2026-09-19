#!/bin/bash -l
#SBATCH --job-name=h1_dual_build
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h1_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h1_%j.err

# H1 TEST: does ONE HIP source compile and run on BOTH backends?
#
# This job runs on an AMD/MI210 node and builds+executes the probe with ROCm.
# A paired job (h1_cuda.sbatch) does the same on an NVIDIA node. Both must
# PASS for the pure-HIP architecture to be confirmed.
#
# Shebang is '#!/bin/bash -l' because the non-interactive Slurm shell does not
# initialise Lmod (learned the hard way on 2026-09-10).
#
# NOTE: no 'set -u' -- the conda/module hooks read unset vars.

set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
MODROOT=/kuhpc/sw/lmod/9.3
[ -f "$MODROOT/init/bash" ] && source "$MODROOT/init/bash"

echo "=== host: $(hostname) ==="
echo "=== platform: AMD / ROCm (requested MI210) ==="

ml purge
# cmake 3.30.3 needs the gcc/14.2 libstdc++ (GLIBCXX_3.4.32);
# load the compiler BEFORE cmake, and cmake BEFORE the gpu toolchain.
ml load compiler/gcc/14.2
ml load cmake/3.30.3
ml load rocm/6.4.3
echo "ROCM_PATH=$ROCM_PATH"
echo "hipcc: $(command -v hipcc)"
hipcc --version | head -2

echo
echo "=== build ==="
cd "$ROOT"
bash scripts/build_rocm.sh 2>&1 | tail -40
BUILD_RC=${PIPESTATUS[0]}

echo
echo "=== H1 RESULT (rocm) ==="
if [ "$BUILD_RC" -eq 0 ]; then
    echo "H1_ROCM: PASS"
else
    echo "H1_ROCM: FAIL (rc=$BUILD_RC)"
fi
exit "$BUILD_RC"
