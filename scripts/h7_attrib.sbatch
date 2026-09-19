#!/bin/bash -l
#SBATCH --job-name=h7_attrib
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h7_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h7_%j.err

# Fase 7 — WHERE does the kernel spend its time?
#
# Before optimising anything, attribute the cost. The kernel has two plausible
# bottlenecks and they imply opposite fixes:
#
#   (A) BARRIERS / DIVERGENCE: one block per pair, blockDim=2*smax+1, and wavefront
#       s only uses 2s+1 of those threads. Useful threads = sum(2s+1) = smax^2,
#       offered = smax*(2*smax) = 2*smax^2, so utilisation is ~25% BY ARITHMETIC.
#       Plus 3 __syncthreads() per wavefront. Fix: fewer, longer-running warps, or
#       multiple pairs per block to fill the idle lanes. Time would scale with
#       smax (barrier count), roughly independent of sequence length.
#
#   (B) CHARACTER EXTENSION: the inner while loop compares pattern[v] == text[h]
#       byte by byte. Work scales with the number of matching characters, i.e.
#       with sequence LENGTH. Fix: vectorised/word-at-a-time extension. Time
#       would scale with len and be insensitive to smax.
#
# The two controls below separate them by varying ONE thing at a time:
#   1. len sweep at FIXED smax   -> isolates extension (bytes) cost
#   2. smax sweep at FIXED len   -> isolates barrier cost
# pairs is adjusted so total cells are comparable, so the comparison is not an
# artefact of doing more work.
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
echo "=== H7: kernel cost attribution (MI210) ==="

/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. \
    -DGENOALIGNER_BENCH_BACKEND='"rocm"' -o "$BIN" bench/tcus.cpp
RC_BUILD=$?
echo "BUILD_RC=$RC_BUILD"
[ "$RC_BUILD" -ne 0 ] && { echo "H7: BUILD FAILED"; exit 1; }

# Only the kernel number matters here; reps high enough to bury launch noise.
probe() {
    local lab="$1"; shift
    echo
    echo "######## $lab"
    "$BIN" "$@" --reps 30 2>&1 | grep -E 'device  :|pairs   :|launch  :|KERNEL \(mean|kernel-only|resolved /|OK:|VERIFY|MISMATCH|---'
}

echo
echo "=================================================================="
echo "CONTROL 1 — len sweep at FIXED smax (isolates EXTENSION / bytes)"
echo "  smax=64 fixed. ident fixed. pairs scaled so cells are ~constant."
echo "=================================================================="
# cells = pairs * len^2 -> hold cells ~2e9: pairs = 2e9/len^2
probe "len=256,  pairs=32000, smax=64"  --len 256  --pairs 32000 --ident 90 --smax 64
probe "len=512,  pairs=8000,  smax=64"  --len 512  --pairs 8000  --ident 90 --smax 64
probe "len=1024, pairs=2000,  smax=64"  --len 1024 --pairs 2000  --ident 90 --smax 64
probe "len=2048, pairs=500,   smax=64"  --len 2048 --pairs 500   --ident 90 --smax 64

echo
echo "=================================================================="
echo "CONTROL 2 — smax sweep at FIXED len (isolates BARRIERS)"
echo "  len=1024, pairs=2000 fixed. smax varies 32..256."
echo "=================================================================="
probe "len=1024, pairs=2000, smax=32"   --len 1024 --pairs 2000 --ident 90 --smax 32
probe "len=1024, pairs=2000, smax=64"   --len 1024 --pairs 2000 --ident 90 --smax 64
probe "len=1024, pairs=2000, smax=128"  --len 1024 --pairs 2000 --ident 90 --smax 128
probe "len=1024, pairs=2000, smax=256"  --len 1024 --pairs 2000 --ident 90 --smax 256

echo
echo "=== H7 DONE ==="
