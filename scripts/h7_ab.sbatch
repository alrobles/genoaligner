#!/bin/bash -l
#SBATCH --job-name=h7_ab
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h7ab_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h7ab_%j.err

# Fase 7, Paso 1 (definitivo) — INTERLEAVED A/B, the only design that survives
# node variance.
#
# What the last three jobs established, in order:
#   29210734: flat was +-0.1% reproducible and 1.06-1.93x faster.
#   29213840: the SAME flat config measured 16.4-20.3 ms on another node where
#             29210734 had measured 5.3 ms -- 3-4x across jobs, with intra-job
#             spread blowing up from +-0.1% to 1.9x.
#
# So node-to-node variance (3-4x) DOMINATES every effect I have been trying to
# measure. Any speedup computed across jobs -- including the 1.93x I reported --
# is arithmetic on a moving baseline. Retracting it here rather than defending it.
#
# The fix is the experimental design, not more analysis: run BOTH kernels in the
# SAME job, ALTERNATING (A,B,A,B,...), so any node-level drift affects both equally
# and cancels in the ratio. The metric reported is the per-round ratio, which is
# valid even when the absolute milliseconds are not comparable across jobs.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
OUT_DIR=/beegfs/a474r867/genoaligner/bench
mkdir -p "$OUT_DIR"

source /kuhpc/sw/lmod/9.3/init/bash 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1
cd "$ROOT" || exit 1

echo "=== host: $(hostname) ==="
echo "=== H7-AB: interleaved comparison, both kernels in one job ==="
echo

/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. -DWFA_FLAT_BLOCK=128 \
    -DGENOALIGNER_BENCH_BACKEND='"rocm"' -o /tmp/tab bench/tcus.cpp
[ $? -ne 0 ] && { echo "BUILD FAILED"; exit 1; }

kms() {   # kernel ms for a given kernel/regime
    /tmp/tab "$@" --reps 30 --verify 5 2>&1 \
        | grep -E 'KERNEL \(mean' | sed 's/.*: *//; s/  <--.*//' | awk '{print $1}'
}

echo "Rounds alternate default/flat at the SAME smax, so node drift hits both."
echo "The RATIO per round is the measurement; absolute ms are not comparable."
echo

for SM in 64 128 256; do
    echo
    echo "######## smax=$SM  (len=1024 pairs=2000 ident=98)"
    printf "%6s %12s %12s %10s\n" "round" "default_ms" "flat_ms" "ratio"
    for R in 1 2 3 4 5; do
        D=$(kms --len 1024 --pairs 2000 --ident 98 --smax $SM --kernel default)
        F=$(kms --len 1024 --pairs 2000 --ident 98 --smax $SM --kernel flat)
        RATIO=$(awk -v d="$D" -v f="$F" 'BEGIN{ if (f>0) printf "%.3f", d/f; else print "n/a" }')
        printf "%6d %12s %12s %10s\n" "$R" "$D" "$F" "$RATIO"
    done
done

echo
echo "=== H7-AB DONE ==="
echo "A ratio column that stays consistent across rounds, while the ms columns drift,"
echo "is the result: the ratio cancels node noise and is the only defensible claim."
