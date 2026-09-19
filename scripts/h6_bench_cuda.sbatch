#!/bin/bash -l
#SBATCH --job-name=h6_bench_cuda
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:q6000:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h6cuda_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h6cuda_%j.err

# Fase 6 on the NVIDIA flank: the SAME bench/tcus.cpp built with nvcc, so the
# comparison between backends is a comparison of the toolchain and the silicon,
# not of two different harnesses.
#
# This is the measurement behind masterplan hypothesis H6 ("the abstraction costs
# nothing"): same source, same regimes, two vendors. Until it runs, that claim is
# asserted, not measured.
#
# CUDA 12.4, not 13.0 -- ROCm's nvidia_detail layer targets 12.x. Same constraint
# as every other CUDA job in this repo.
set -o pipefail

R=/kuhpc/sw/rocm/6.4.3
C=/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4
ROOT=/beegfs/a474r867/genoaligner/repo
OUT_DIR=/beegfs/a474r867/genoaligner/bench
mkdir -p "$OUT_DIR"
BIN=/tmp/tcus_cuda

echo "=== host: $(hostname) ==="
echo "=== H6-CUDA: Fase 6 phase-separated timing probe (NVIDIA) ==="
echo

nvidia-smi -L 2>/dev/null || echo "no nvidia-smi"
echo "--- nvcc ---"
"$C/bin/nvcc" --version | tail -2

cd "$ROOT" || exit 1

echo
echo "--- build (nvcc, __HIP_PLATFORM_NVIDIA__) ---"
"$C/bin/nvcc" -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
    -gencode arch=compute_75,code=sm_75 \
    -DGENOALIGNER_BENCH_BACKEND='"cuda"' \
    -I "$R/include" \
    -I "$C/include" \
    -I "$C/targets/x86_64-linux/include" \
    -I . \
    -lcuda -L "$C/lib64/stubs" \
    -x cu -o "$BIN" bench/tcus.cpp
RC_BUILD=$?
echo "BUILD_RC=$RC_BUILD"
if [ "$RC_BUILD" -ne 0 ]; then
    echo "H6-CUDA: BUILD FAILED"
    exit 1
fi

echo
echo "--- embedded GPU code ---"
"$C/bin/cuobjdump" --list-elf "$BIN" 2>/dev/null | head -3

rc_all=0
rc_verified=0
run() {
    local label="$1"; shift
    echo
    echo "=============================================================="
    echo "REGIME: $label"
    echo "=============================================================="
    "$BIN" "$@" | tee -a "$OUT_DIR/h6_cuda_${SLURM_JOB_ID}.txt"
    local rc=${PIPESTATUS[0]}
    case "$rc" in
        0) : ;;
        5) echo ">>> regime UNVERIFIED (rc=5): timing measured nothing. Not a result."; rc_all=1 ;;
        6) echo ">>> regime INVALID (rc=6): kernel wrong. Not a result."; rc_all=1 ;;
        *) echo ">>> regime FAILED (rc=$rc)."; rc_all=1 ;;
    esac
    [ "$rc" -eq 0 ] && rc_verified=$((rc_verified + 1))
    return 0
}

# Same regimes as the ROCm job, so the two logs are directly comparable.
run "len=256,  smax=64,  8000 pairs, ident=90%"  --len 256  --pairs 8000 --ident 90 --smax 64  --reps 20
run "len=1024, smax=64,  2000 pairs, ident=90%"  --len 1024 --pairs 2000 --ident 90 --smax 64  --reps 20
run "len=1024, smax=256, 2000 pairs, ident=90%"  --len 1024 --pairs 2000 --ident 90 --smax 256 --reps 20
run "len=1024, smax=256, 2000 pairs, ident=70%"  --len 1024 --pairs 2000 --ident 70 --smax 256 --reps 20

echo
echo "=== H6-CUDA RESULT ==="
echo "  regimes verified and timed : $rc_verified"
if [ "$rc_all" -eq 0 ]; then echo "H6-CUDA: PASS"; else echo "H6-CUDA: FAIL (see the >>> lines above)"; fi
exit "$rc_all"
