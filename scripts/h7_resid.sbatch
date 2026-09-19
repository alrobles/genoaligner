#!/bin/bash -l
#SBATCH --job-name=h7_resid
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h7resid_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h7resid_%j.err

# Fase 7, Paso 1 — close the flat kernel's residual smax scaling.
#
# The flat kernel is stable (±0.1%) but still goes 1.77 -> 5.33 ms from smax=64 to
# 256 at CONSTANT real distance (~15). Four hypotheses are already refuted. The one
# thing that remains and scales with smax in BOTH kernels is the shared-memory
# footprint: 2*(2*smax+3) ints, i.e. 0.5 KB at smax=64 vs 4 KB at smax=256.
#
# The clean discriminator: hold smax FIXED and vary the real distance. If cost per
# wavefront is what scales with smax, then at smax=256 a pair that walks 230
# wavefronts must cost ~15x a pair that walks 15 -- and the two must land on the
# SAME time-per-wavefront line. If instead the whole kernel gets slower with smem
# regardless of how many wavefronts it walks, the fixed cost per launch scales with
# smax and the growing smem is implicated.
#
# Both kernels run in the SAME job so the comparison never crosses nodes.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
OUT_DIR=/beegfs/a474r867/genoaligner/bench
mkdir -p "$OUT_DIR"

source /kuhpc/sw/lmod/9.3/init/bash 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1
cd "$ROOT" || exit 1

echo "=== host: $(hostname) ==="
echo "=== H7-resid: what scales with smax? ==="
echo

/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. -DWFA_FLAT_BLOCK=128 \
    -DGENOALIGNER_BENCH_BACKEND='"rocm"' -o /tmp/tresid bench/tcus.cpp
[ $? -ne 0 ] && { echo "BUILD FAILED"; exit 1; }

probe() {
    local lab="$1"; shift
    printf "%-42s" "$lab"
    /tmp/tresid "$@" --reps 30 --kernel flat --verify 5 2>&1 \
        | grep -E 'KERNEL \(mean|distance \(score|resolved /|occupancy|launch  :' \
        | sed 's/.*: *//; s/  <--.*//' | tr '\n' ' '
    echo
}

echo "TEST A — smax FIXED at 256, real distance swept 15 (never walks far)"
echo "  if cost/wavefront explains it, time must scale with distance here"
echo
printf "%-42s %s\n" "config" "kernel_ms  dist_mean"
probe "smax=256 ident=98 (dist~15)"   --len 1024 --pairs 2000 --ident 98 --smax 256
probe "smax=256 ident=90 (dist~100)"  --len 1024 --pairs 2000 --ident 90 --smax 256
probe "smax=256 ident=80 (dist~200)"  --len 1024 --pairs 2000 --ident 80 --smax 256

echo
echo "TEST B — real distance FIXED low, smax swept (the residual in question)"
echo "  if smem/launch dominates, time scales with smax even at constant distance"
echo
probe "smax= 64 ident=99 (dist~10)"   --len 1024 --pairs 2000 --ident 99 --smax 64
probe "smax=128 ident=99 (dist~10)"   --len 1024 --pairs 2000 --ident 99 --smax 128
probe "smax=256 ident=99 (dist~10)"   --len 1024 --pairs 2000 --ident 99 --smax 256
probe "smax=256 ident=99 (dist~10) again" --len 1024 --pairs 2000 --ident 99 --smax 256

echo
echo "TEST C — same smax, same distance, half the pairs (per-pair vs per-launch)"
probe "smax=256 ident=99 pairs=2000"  --len 1024 --pairs 2000 --ident 99 --smax 256
probe "smax=256 ident=99 pairs=1000"  --len 1024 --pairs 1000 --ident 99 --smax 256

echo
echo "TEST D — the SAME configuration 4x, to establish the flat kernel's spread here"
echo "  job 29210734 measured +-0.1% for the flat; this job saw 15.807 vs 10.779 ms"
echo "  (47%) for one identical config, so the spread has to be re-established."
echo
printf "%-42s %s\n" "config" "kernel_ms dist resolved"
probe "rep1 smax=256 ident=99 pairs=2000" --len 1024 --pairs 2000 --ident 99 --smax 256
probe "rep2 smax=256 ident=99 pairs=2000" --len 1024 --pairs 2000 --ident 99 --smax 256
probe "rep3 smax=256 ident=99 pairs=2000" --len 1024 --pairs 2000 --ident 99 --smax 256
probe "rep4 smax=256 ident=99 pairs=2000" --len 1024 --pairs 2000 --ident 99 --smax 256

echo
echo "=== H7-resid DONE ==="
