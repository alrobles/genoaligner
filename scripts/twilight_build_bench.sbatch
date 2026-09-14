#!/bin/bash -l
#SBATCH --job-name=twilight_bb
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/twilight_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/twilight_%j.err

# Build TWILIGHT (CUDA path) on the cluster and smoke-test it.
set -o pipefail
TW=/beegfs/a474r867/twilight
LOG=/beegfs/a474r867/genoaligner/logs
C=/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4
BOOST=/kuhpc/sw/boost/1.87.0/gcc/11.4
TBBDIR=/kuhpc/sw/tbb/2022/gcc/11.4/lib64/cmake/TBB
CMAKE=/kuhpc/sw/cmake/3.30.3/gcc/11.4/bin/cmake

echo "=== host: $(hostname) $(date) ==="
nvidia-smi -L 2>/dev/null || true
gcc --version | head -1
"$C/bin/nvcc" --version | tail -1

cd "$TW" || exit 1
mkdir -p build && cd build
"$CMAKE" .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$C/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_CUDA_FLAGS="-D_AMXTILEINTRIN_H_INCLUDED -D_AMXBF16INTRIN_H_INCLUDED -D_AMXINT8INTRIN_H_INCLUDED" \
    -DBOOST_ROOT="$BOOST" \
    -DBoost_NO_SYSTEM_PATHS=ON \
    -DTBB_DIR="$TBBDIR" \
    2>&1 | tail -15
"$CMAKE" --build . -j 8 2>&1 | tail -20

BIN=$(find . -name twilight -type f | head -1)
[ -x "$BIN" ] || { echo "BUILD FAILED - no binary"; exit 1; }
echo "binary: $BIN"
cp "$BIN" "$TW/twilight-v100"
"$TW/twilight-v100" --help 2>&1 | head -30
echo "TWILIGHT-BUILD: PASS"
