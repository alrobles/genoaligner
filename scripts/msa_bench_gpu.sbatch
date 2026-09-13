#!/bin/bash -l
#SBATCH --job-name=genomsa_bench_gpu
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/msabenchgpu_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/msabenchgpu_%j.err
# submit with:  sbatch --gres=gpu:mi210:1 scripts/msa_bench_gpu.sbatch
#            or sbatch --gres=gpu:v100:1  --export=TAG=v100 ...
#
# genomsa GPU path on the bench1.0 sets. Per-family wall clock in
# $BASE/genomsa_<TAG>_times.tsv; alignments in $BASE/genomsa_<TAG>_out/.
set -o pipefail
REPO=/beegfs/a474r867/genoaligner/repo
BENCH=/beegfs/a474r867/bench/bench1.0
BASE=/beegfs/a474r867/bench
TAG=${TAG:-mi210}
OUT=$BASE/genomsa_${TAG}_out
TIMES=$BASE/genomsa_${TAG}_times.tsv
cd "$REPO" || exit 1
echo "=== host: $(hostname) tag=$TAG $(date) ==="
rocm-smi --showproductname 2>/dev/null | head -3 || nvidia-smi --query-gpu=name --format=csv 2>/dev/null | head -3

if [ "$TAG" = "v100" ]; then
    R=/kuhpc/sw/rocm/6.4.3
    C=/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4
    "$C/bin/nvcc" -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
        -gencode arch=compute_70,code=sm_70 \
        -I "$R/include" -I "$C/include" -I "$C/targets/x86_64-linux/include" \
        -I . -I include -lcuda -L "$C/lib64/stubs" -x cu \
        -o /tmp/genomsa_gpu tools/genomsa.cpp src/msa/msa_gpu.cpp \
        src/msa/msa_ref.cpp || { echo BUILD_FAILED; exit 1; }
else
    /kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -ffp-contract=off -std=c++17 -I. -Iinclude \
        -o /tmp/genomsa_gpu tools/genomsa.cpp src/msa/msa_gpu.cpp \
        src/msa/msa_ref.cpp || { echo BUILD_FAILED; exit 1; }
fi
echo "build ok"

run_one() {
    d=$1; f=$2
    in="$BENCH/$d/in/$f"; out="$OUT/$d/$f"
    s=$EPOCHREALTIME
    /tmp/genomsa_gpu "$in" "$out" --protein >/dev/null 2>&1
    rc=$?
    e=$EPOCHREALTIME
    ok=ok; { [ $rc -ne 0 ] || [ ! -s "$out" ]; } && ok=FAIL
    printf "genomsa_%s\t%s\t%s\t%.3f\t%s\n" "$TAG" "$d" "$f" \
        "$(echo "$e - $s" | bc)" "$ok" >> "$TIMES"
}
export -f run_one
export BENCH OUT TIMES TAG

for d in bali3 sabrem ox prefab4; do
    mkdir -p "$OUT/$d"
    echo "=== $d ==="
    ls "$BENCH/$d/in" | xargs -P 8 -I{} bash -c 'run_one "$0" "$1"' "$d" {}
done
echo "BENCH-GPU DONE tag=$TAG"
