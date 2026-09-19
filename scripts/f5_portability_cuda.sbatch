#!/bin/bash -l
#SBATCH --job-name=f5_port_cuda
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:q6000:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/f5cuda_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/f5cuda_%j.err

# FASE 5, NVIDIA side: build with nvcc over the SAME source and run on a GPU.
# Companion to f5_portability.sbatch (MI210).
#
# Both jobs exercise the portability layer end to end: one CMakeLists, one
# source tree, two compilers. The fusion criterion for Fase 5 is that the tests
# from Fases 2-4 pass on BOTH backends from the same source; this job covers the
# build-and-run half for NVIDIA.
set -u

export LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64:${LD_LIBRARY_PATH:-}
CMAKE=/kuhpc/sw/cmake/3.30.3/gcc/14.2/bin/cmake
ROCM=/kuhpc/sw/rocm/6.4.3
CUDA=/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4
ROOT=/beegfs/a474r867/genoaligner/repo
BUILD=/beegfs/a474r867/genoaligner/build_f5_cuda

echo "=== host: $(hostname) ==="
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi"
echo "=== Fase 5: CUDA build + run on q6000 (nvcc, same source) ==="

source /kuhpc/sw/lmod/9.3/init/bash 2>/dev/null
ml purge >/dev/null 2>&1
ml load compiler/gcc/14.2 >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1   # for the HIP header layer, not the compiler

cd "$ROOT" || exit 1
rm -rf "$BUILD"

echo "--- configure (backend=cuda, nvcc) ---"
$CMAKE -S . -B "$BUILD" \
    -DGENOALIGNER_BACKEND=cuda \
    -DCMAKE_CXX_COMPILER=$CUDA/bin/nvcc \
    -DGENOALIGNER_CUDA_ROOT=$CUDA \
    -DGENOALIGNER_ROCM_ROOT=$ROCM \
    -DCMAKE_CXX_FLAGS="-w -gencode arch=compute_75,code=sm_75" \
    -DCMAKE_BUILD_TYPE=Release
RC=$?
[ "$RC" -eq 0 ] || { echo "CONFIGURE FAILED"; exit 1; }

echo "--- build ---"
$CMAKE --build "$BUILD" -j 8
RC=$?
[ "$RC" -eq 0 ] || { echo "BUILD FAILED"; exit 1; }

echo "--- embedded GPU code ---"
$CUDA/bin/cuobjdump --list-elf "$BUILD/genoaligner_probe" 2>/dev/null | head -3

echo "--- run probe on q6000 ---"
"$BUILD/genoaligner_probe"
RC=$?

echo
echo "=== FASE 5 CUDA RESULT ==="
if [ "$RC" -eq 0 ]; then echo "FASE 5 CUDA: PASS"; else echo "FASE 5 CUDA: FAIL (rc=$RC)"; fi
exit "$RC"
