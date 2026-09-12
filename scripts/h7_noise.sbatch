#!/bin/bash -l
#SBATCH --job-name=h7_noise
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h7noise_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h7noise_%j.err

# Fase 7 — how much of the "scaling" is just measurement noise?
#
# The flat kernel measured 6.286 ms at smax=256 in job 29210729 and 5.355 ms in
# job 29210731 -- same binary, same input, same node type, 17% apart. If runs on
# different nodes vary that much, then some of the residual smax scaling I have
# been trying to explain may not be a kernel property at all.
#
# This runs the SAME configuration several times IN ONE JOB and reports the spread.
# Two questions answered at once:
#   1. within-job variance (the floor for any comparison), and
#   2. whether the smax trend survives when the same process measures all points,
#      which removes cross-node variation from the comparison entirely.
#
# Cross-node variance is not measurable from a single job and is NOT claimed here.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
OUT_DIR=/beegfs/a474r867/genoaligner/bench
mkdir -p "$OUT_DIR"

source /kuhpc/sw/lmod/9.3/init/bash 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1
cd "$ROOT" || exit 1

echo "=== host: $(hostname) ==="
echo "=== H7-noise: repeatability of the smax sweep ==="
echo

/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. -DWFA_FLAT_BLOCK=128 \
    -DGENOALIGNER_BENCH_BACKEND='"rocm"' -o /tmp/tnoise bench/tcus.cpp
[ $? -ne 0 ] && { echo "BUILD FAILED"; exit 1; }

echo "--- same regime, 5 repeats in ONE process, flat kernel ---"
for SM in 64 128 256; do
    echo
    echo "######## smax=$SM"
    for R in 1 2 3 4 5; do
        printf "  rep%d: " "$R"
        /tmp/tnoise --len 1024 --pairs 2000 --ident 98 --smax $SM --reps 30 \
            --kernel flat --verify 5 2>&1 \
            | grep -E 'KERNEL \(mean' | sed 's/.*: *//'
    done
done

echo
echo "--- the original kernel, same regime, for a same-job comparison ---"
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. \
    -DGENOALIGNER_BENCH_BACKEND='"rocm"' -o /tmp/tnoise_base bench/tcus.cpp
for SM in 64 128 256; do
    echo
    echo "######## default smax=$SM"
    for R in 1 2 3; do
        printf "  rep%d: " "$R"
        /tmp/tnoise_base --len 1024 --pairs 2000 --ident 98 --smax $SM --reps 30 \
            --kernel default --verify 5 2>&1 \
            | grep -E 'KERNEL \(mean' | sed 's/.*: *//'
    done
done

echo
echo "=== H7-noise DONE ==="
