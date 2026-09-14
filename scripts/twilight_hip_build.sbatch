#!/bin/bash -l
#SBATCH --job-name=tw_hip
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/twhip_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/twhip_%j.err

# Build TWILIGHT's HIP port on MI210 and run the same sim64 benchmark,
# watching for the silent "CPU on No. X" fallback that papers over kernel
# span errors (present only in the HIP source, commented out in CUDA).
set -o pipefail
TW=/beegfs/a474r867/twilight
B=/beegfs/a474r867/twilight_bench
ROCM=/kuhpc/sw/rocm/6.4.3
BOOST=/kuhpc/sw/boost/1.87.0/gcc/11.4
TBBDIR=/kuhpc/sw/tbb/2022/gcc/11.4/lib64/cmake/TBB
CMAKE=/kuhpc/sw/cmake/3.30.3/gcc/11.4/bin/cmake

echo "=== host: $(hostname) $(date) ==="
"$ROCM/bin/rocm-smi" --showproductname 2>/dev/null | head -3

cd "$TW" || exit 1
rm -rf build-hip && mkdir build-hip && cd build-hip
"$CMAKE" .. \
    -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
    -DHIP_ROOT_DIR="$ROCM" \
    -DBOOST_ROOT="$BOOST" -DBoost_NO_SYSTEM_PATHS=ON \
    -DTBB_DIR="$TBBDIR" \
    2>&1 | tail -10
"$CMAKE" --build . -j 8 2>&1 | tail -15

BIN=$(find "$TW" "$PWD" -name twilight -type f -newer "$TW/CMakeLists.txt" 2>/dev/null | head -1)
[ -x "$BIN" ] || { echo "HIP BUILD FAILED"; exit 1; }
cp "$BIN" "$TW/twilight-mi210"
echo "binary: $TW/twilight-mi210"
"$TW/twilight-mi210" --version 2>&1 | head -2

cd "$B" || exit 1
echo; echo "### sim64 with NJ tree (HIP, MI210) ###"
"$TW/twilight-mi210" -t sim64_njtree.nwk -i sim64_in.fasta -o tw_hip_sim64.aln --type n --overwrite 2>&1 | tee /tmp/tw_hip.log | tail -8
echo "--- CPU-fallback count ---"
grep -c "CPU on No" /tmp/tw_hip.log || true

echo; echo "### COI (HIP, MI210) ###"
"$TW/twilight-mi210" -t "$B/COI.nwk" -i /beegfs/a474r867/phylogenyAI/data/genes_qc_pass/COI.fasta -o "$B/COI.tw_hip.aln" --type n --overwrite 2>&1 | tee /tmp/tw_hip_coi.log | tail -6
echo "--- CPU-fallback count (COI) ---"
grep -c "CPU on No" /tmp/tw_hip_coi.log || true
echo "TW-HIP: DONE"
