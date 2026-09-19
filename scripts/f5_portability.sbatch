#!/bin/bash -l
#SBATCH --job-name=f5_portability
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/f5_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/f5_%j.err

# FASE 5: the portability layer must build and RUN on real hardware.
#
# The login-node test proves both backends configure and compile from one source.
# This proves the ROCm artifact actually executes on the MI210, which is the
# half that compile-only checks cannot cover.
#
# The NVIDIA half runs in the companion job f5_portability_cuda.sbatch (q6000).
set -u

export LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64:${LD_LIBRARY_PATH:-}
CMAKE=/kuhpc/sw/cmake/3.30.3/gcc/14.2/bin/cmake
ROCM=/kuhpc/sw/rocm/6.4.3
ROOT=/beegfs/a474r867/genoaligner/repo
BUILD=/beegfs/a474r867/genoaligner/build_f5_rocm

echo "=== host: $(hostname) ==="
echo "=== Fase 5: ROCm build + run on MI210 ==="

source /kuhpc/sw/lmod/9.3/init/bash 2>/dev/null
ml purge >/dev/null 2>&1
ml load compiler/gcc/14.2 >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1

cd "$ROOT" || exit 1
rm -rf "$BUILD"

echo "--- configure (backend=rocm) ---"
$CMAKE -S . -B "$BUILD" \
    -DGENOALIGNER_BACKEND=rocm \
    -DCMAKE_CXX_COMPILER=$ROCM/bin/hipcc \
    -DCMAKE_BUILD_TYPE=Release
RC=$?
[ "$RC" -eq 0 ] || { echo "CONFIGURE FAILED"; exit 1; }

echo "--- build ---"
$CMAKE --build "$BUILD" -j 8
RC=$?
[ "$RC" -eq 0 ] || { echo "BUILD FAILED"; exit 1; }

echo "--- run probe on MI210 ---"
"$BUILD/genoaligner_probe"
RC=$?

echo
echo "=== FASE 5 ROCM RESULT ==="
if [ "$RC" -eq 0 ]; then echo "FASE 5 ROCM: PASS"; else echo "FASE 5 ROCM: FAIL (rc=$RC)"; fi
exit "$RC"
