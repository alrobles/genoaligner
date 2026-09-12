#!/bin/bash -l
#SBATCH --job-name=h6_bench_a100
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h6a100_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h6a100_%j.err

# Fase 6 on the A100 (sm_80). Same bench/tcus.cpp, same regimes, CUDA 12.4 --
# the toolchain every other CUDA job in this repo uses, and the one the ROCm
# nvidia_detail layer supports.
#
# See scripts/h6_bench_cuda.sbatch for the q6000 (sm_75) twin. The only
# differences are --gres, and the -gencode target.
set -o pipefail

R=/kuhpc/sw/rocm/6.4.3
C=/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4
ARCH=${ARCH:-compute_80,code=sm_80}
ROOT=/beegfs/a474r867/genoaligner/repo
OUT_DIR=/beegfs/a474r867/genoaligner/bench
mkdir -p "$OUT_DIR"
BIN=/tmp/tcus_a100

echo "=== host: $(hostname) ==="
echo "=== H6-A100: Fase 6 timing probe (A100, sm_80) ==="
echo

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
"$C/bin/nvcc" --version | tail -2

cd "$ROOT" || exit 1

echo
echo "--- build (nvcc 12.4, $ARCH) ---"
"$C/bin/nvcc" -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
    -gencode arch=$ARCH \
    -DGENOALIGNER_BENCH_BACKEND='"cuda"' \
    -I "$R/include" \
    -I "$C/include" \
    -I "$C/targets/x86_64-linux/include" \
    -I . \
    -lcuda -L "$C/lib64/stubs" \
    -x cu -o "$BIN" bench/tcus.cpp
RC_BUILD=$?
echo "BUILD_RC=$RC_BUILD"
if [ "$RC_BUILD" -ne 0 ]; then echo "H6-A100: BUILD FAILED"; exit 1; fi

"$C/bin/cuobjdump" --list-elf "$BIN" 2>/dev/null | head -3

rc_all=0
rc_verified=0
run() {
    local label="$1"; shift
    echo
    echo "=============================================================="
    echo "REGIME: $label"
    echo "=============================================================="
    "$BIN" "$@" | tee -a "$OUT_DIR/h6_a100_${SLURM_JOB_ID}.txt"
    local rc=${PIPESTATUS[0]}
    case "$rc" in
        0) : ;;
        5) echo ">>> regime UNVERIFIED (rc=5): timing measured nothing."; rc_all=1 ;;
        6) echo ">>> regime INVALID (rc=6): kernel wrong."; rc_all=1 ;;
        *) echo ">>> regime FAILED (rc=$rc)."; rc_all=1 ;;
    esac
    [ "$rc" -eq 0 ] && rc_verified=$((rc_verified + 1))
    return 0
}

# Identical regimes to the MI210 and q6000 jobs so the three logs are directly
# comparable. Same order, same flags -- a different order would invite a
# comparison-of-typos.
run "len=256,  smax=64,  8000 pairs, ident=90%"  --len 256  --pairs 8000 --ident 90 --smax 64  --reps 20
run "len=1024, smax=64,  2000 pairs, ident=90%"  --len 1024 --pairs 2000 --ident 90 --smax 64  --reps 20
run "len=1024, smax=256, 2000 pairs, ident=90%"  --len 1024 --pairs 2000 --ident 90 --smax 256 --reps 20
run "len=1024, smax=256, 2000 pairs, ident=70%"  --len 1024 --pairs 2000 --ident 70 --smax 256 --reps 20

echo
echo "=== H6-A100 RESULT ==="
echo "  regimes verified and timed : $rc_verified"
if [ "$rc_all" -eq 0 ]; then echo "H6-A100: PASS"; else echo "H6-A100: FAIL"; fi
exit "$rc_all"
