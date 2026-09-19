#!/bin/bash -l
#SBATCH --job-name=h7_attrib2
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h7b_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h7b_%j.err

# Fase 7 — cost attribution, TAKE 2.
#
# Take 1 (job 29210721) failed its own premise: at ident=90% the true edit
# distance exceeds smax whenever len grows or smax shrinks, so most regimes
# ABANDONED their pairs and were timing the guard, not the kernel. Only two rows
# of that run were valid, and the two were not comparable to each other.
#
# The fix is to keep EVERY pair resolvable in EVERY regime, so a difference in
# time is a difference in work and not in abandonment. Identity is therefore a
# function of the regime, not a constant:
#
#   expected substitutions ~ len*(1-ident) plus a couple of indels; keep that
#   comfortably under smax. So ident rises with len and falls with smax.
#
# Every probe below states its expected distance and the run PRINTS the abandoned
# count, so a bad assumption shows up as >>> rather than as a number.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
OUT_DIR=/beegfs/a474r867/genoaligner/bench
mkdir -p "$OUT_DIR"
BIN=/tmp/tcus_attrib

source /kuhpc/sw/lmod/9.3/init/bash 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1
cd "$ROOT" || exit 1

echo "=== host: $(hostname) ==="
echo "=== H7-b: kernel cost attribution, all pairs resolvable (MI210) ==="

/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. \
    -DGENOALIGNER_BENCH_BACKEND='"rocm"' -o "$BIN" bench/tcus.cpp
RC_BUILD=$?
echo "BUILD_RC=$RC_BUILD"
[ "$RC_BUILD" -ne 0 ] && { echo "H7-b: BUILD FAILED"; exit 1; }

probe() {
    local lab="$1"; shift
    echo
    echo "######## $lab"
    "$BIN" "$@" --reps 30 2>&1 | grep -E 'pairs   :|KERNEL \(mean|kernel-only|resolved /|distance \(score|wavefronts walked|\^ \*\*|pairs with d|OK:|MISMATCH|>>>'
}

echo
echo "=================================================================="
echo "CONTROL 1 — len sweep at FIXED smax=128 (isolates EXTENSION / bytes)"
echo "  ident tuned per len so expected distance (~len/100) stays < 128."
echo "  pairs scaled as 2e9/len^2 to hold cells ~constant."
echo "=================================================================="
probe "len=256,  pairs=32000, ident=99%, smax=128  (exp dist ~3)"    --len 256  --pairs 32000 --ident 99 --smax 128
probe "len=512,  pairs=8000,  ident=98%, smax=128  (exp dist ~10)"   --len 512  --pairs 8000  --ident 98 --smax 128
probe "len=1024, pairs=2000,  ident=97%, smax=128  (exp dist ~31)"   --len 1024 --pairs 2000  --ident 97 --smax 128
probe "len=2048, pairs=500,   ident=96%, smax=128  (exp dist ~82)"   --len 2048 --pairs 500   --ident 96 --smax 128

echo
echo "=================================================================="
echo "CONTROL 2 — smax sweep at FIXED len=1024, small fixed distance"
echo "  ident=98% -> expected distance ~20, so it resolves at every smax."
echo "  Work per pair is the SAME; only the barrier count changes."
echo "=================================================================="
probe "len=1024, pairs=2000, smax=32,  ident=98%  (exp dist ~20)"   --len 1024 --pairs 2000 --ident 98 --smax 32
probe "len=1024, pairs=2000, smax=64,  ident=98%  (exp dist ~20)"   --len 1024 --pairs 2000 --ident 98 --smax 64
probe "len=1024, pairs=2000, smax=128, ident=98%  (exp dist ~20)"   --len 1024 --pairs 2000 --ident 98 --smax 128
probe "len=1024, pairs=2000, smax=256, ident=98%  (exp dist ~20)"   --len 1024 --pairs 2000 --ident 98 --smax 256

echo
echo "=== H7-b DONE ==="
