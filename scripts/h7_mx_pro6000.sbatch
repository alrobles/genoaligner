#!/bin/bash -l
#SBATCH --job-name=h7_mx_pro6000
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:pro6000:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h7mxp6_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h7mxp6_%j.err

# Fase 7, Paso 2 — re-measure the Fase 6 regimes with the flat kernel, using the
# INTERLEAVED design that the phase's measurements settled on.
#
# Two differences from the Fase 6 matrix, both deliberate:
#
# 1. INTERLEAVED. Fase 6 compared kernels across jobs. Job 29213840 then showed
#    3-4x node-to-node variance on identical input, which invalidates any
#    cross-job comparison. Here both kernels run alternately IN ONE JOB and the
#    per-round ratio is the reported metric.
#
# 2. SPREAD, not a single number. The default kernel was measured oscillating
#    4.4x within one process (2.179-9.619 ms). A mean without its spread hides that,
#    so every row reports both.
#
# The regimes are the ones Fase 6 used, so the comparison is like-for-like:
#   len=256  smax=64   many pairs
#   len=1024 smax=256  the project's phylogenetic regime, two identities
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
OUT_DIR=/beegfs/a474r867/genoaligner/bench
mkdir -p "$OUT_DIR"

cd "$ROOT" || exit 1
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "no nvidia-smi" 

echo "=== host: $(hostname) ==="
echo "=== H7-mx: Fase 6 regimes, interleaved default vs flat, RTX PRO 6000 (sm_120) ==="
echo

R=/kuhpc/sw/rocm/6.4.3
C=/kuhpc/sw/nvhpc/Linux_x86_64/25.3/cuda/12.8    # 12.8: knows sm_120 AND compiles nvidia_detail
"$C/bin/nvcc" -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
    -gencode arch=compute_120,code=sm_120 \
    -DWFA_FLAT_BLOCK=128 -DGENOALIGNER_BENCH_BACKEND='"cuda"' \
    -I "$R/include" -I "$C/include" -I "$C/targets/x86_64-linux/include" -I . \
    -lcuda -L "$C/lib64/stubs" -x cu -o /tmp/tmxp6 bench/tcus.cpp
[ $? -ne 0 ] && { echo "BUILD FAILED"; exit 1; }

# Emits: "kernel_ms ratio_ok". --verify must pass (exit 0) or the row is void.
row() {
    local lab="$1"; shift
    printf "%-46s" "$lab"
    local out
    out=$(/tmp/tmxp6 "$@" --reps 30 --verify 50 2>&1)
    local rc=$?
    local km res
    km=$(echo "$out" | grep -E 'KERNEL \(mean' | sed 's/.*: *//; s/  <--.*//' | awk '{print $1}')
    res=$(echo "$out" | grep -E 'resolved /' | sed 's/.*: *//; s/  (abandoned.*//')
    if [ "$rc" -ne 0 ]; then
        echo "rc=$rc  NOT VERIFIED -- row void"
        return
    fi
    echo "${km} ms   resolved ${res}"
}

echo "Each regime: default and flat, 4 interleaved rounds. The RATIO cancels node drift."
echo

for REGIME in "256 8000 90 64" "1024 2000 90 256" "1024 2000 70 256"; do
    set -- $REGIME
    LEN=$1; PAIRS=$2; IDENT=$3; SMAX=$4
    echo
    echo "######## len=$LEN pairs=$PAIRS ident=$IDENT% smax=$SMAX"
    printf "%6s   %-34s %-34s %8s\n" "round" "default" "flat" "ratio"
    for R in 1 2 3 4; do
        outD=$(/tmp/tmxp6 --len $LEN --pairs $PAIRS --ident $IDENT --smax $SMAX --reps 20 --verify 50 --kernel default 2>&1)
        outF=$(/tmp/tmxp6 --len $LEN --pairs $PAIRS --ident $IDENT --smax $SMAX --reps 20 --verify 50 --kernel flat 2>&1)
        D=$(echo "$outD" | grep -E 'KERNEL \(mean' | sed 's/.*: *//; s/  <--.*//' | awk '{print $1}')
        F=$(echo "$outF" | grep -E 'KERNEL \(mean' | sed 's/.*: *//; s/  <--.*//' | awk '{print $1}')
        # verify must pass on BOTH or the row is meaningless
        okD=$(echo "$outD" | grep -cE 'OK: [0-9]+/[0-9]+ sampled')
        okF=$(echo "$outF" | grep -cE 'OK: [0-9]+/[0-9]+ sampled')
        RATIO=$(awk -v d="$D" -v f="$F" 'BEGIN{ if (f>0) printf "%.3f", d/f; else print "n/a" }')
        printf "%6d   %-34s %-34s %8s%s\n" "$R" "${D} ms" "${F} ms" "$RATIO" \
               "$([ "$okD" = 0 ] || [ "$okF" = 0 ] && echo '   <-- VERIFY FAILED')"
    done
done

echo
echo "=== H7-mx DONE ==="