#!/bin/bash -l
#SBATCH --job-name=h6_bench_pro6000
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:pro6000:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h6pro6000_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h6pro6000_%j.err

# Fase 6 on the RTX PRO 6000 (sm_120, Blackwell). Same bench/tcus.cpp, same
# regimes -- but NOT the same CUDA: sm_120 needs >= 12.8, while every other CUDA job
# in this repo uses 12.4. Found at /kuhpc/sw/nvhpc/Linux_x86_64/25.3/cuda/12.8.
#
# 12.8 is the one version that satisfies BOTH constraints: it knows sm_120 (13.0
# does too, but 13.0 does not compile the ROCm nvidia_detail header layer), and it
# is still 12.x (so the layer compiles). Its existence was briefly missed: an
# earlier conclusion of 'no CUDA 12.8+ on the site' came from checking only
# /kuhpc/sw/cuda-toolkit -- which is 12.4-only -- and the two nvhpc DEFAULT dirs,
# without walking the versioned nvhpc installs. The negative was wrong.
#
# Twins: h6_bench_{cuda,a100,v100,l40}.sbatch -- identical except --gres, CUDA and
# -gencode.
set -o pipefail

R=/kuhpc/sw/rocm/6.4.3
C=/kuhpc/sw/nvhpc/Linux_x86_64/25.3/cuda/12.8
ARCH=${ARCH:-compute_120,code=sm_120}
ROOT=/beegfs/a474r867/genoaligner/repo
OUT_DIR=/beegfs/a474r867/genoaligner/bench
mkdir -p "$OUT_DIR"
BIN=/tmp/tcus_pro6000

echo "=== host: $(hostname) ==="
echo "=== H6-PRO6000: Fase 6 timing probe (RTX PRO 6000, sm_120) ==="
echo

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
"$C/bin/nvcc" --version | tail -2

cd "$ROOT" || exit 1

echo
echo "--- build (nvcc 12.8, $ARCH) ---"
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
if [ "$RC_BUILD" -ne 0 ]; then echo "H6-PRO6000: BUILD FAILED"; exit 1; fi

"$C/bin/cuobjdump" --list-elf "$BIN" 2>/dev/null | head -3

rc_all=0
rc_verified=0
run() {
    local label="$1"; shift
    echo
    echo "=============================================================="
    echo "REGIME: $label"
    echo "=============================================================="
    "$BIN" "$@" | tee -a "$OUT_DIR/h6_pro6000_${SLURM_JOB_ID}.txt"
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
echo "=== H6-PRO6000 RESULT ==="
echo "  regimes verified and timed : $rc_verified"
if [ "$rc_all" -eq 0 ]; then echo "H6-PRO6000: PASS"; else echo "H6-PRO6000: FAIL"; fi
exit "$rc_all"