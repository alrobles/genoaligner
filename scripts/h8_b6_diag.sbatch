#!/bin/bash -l
#SBATCH --job-name=h8_b6diag
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h8b6d_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h8b6d_%j.err

# B6 diagnostic — why is one rep in seven reporting ~half the throughput?
#
# Observed in job 29226416, regime len=1024 smax=256 ident=90%:
#   default: 0.421 x6  and 0.212 once   (0.212 is almost exactly 0.421/2)
#   flat:    1.118 1.112 1.117 [0.721] 1.119 1.121 1.119
#
# The TCUPS the bench reports is computed over the cells of RESOLVED pairs. A run in
# which half the pairs are abandoned therefore reports roughly HALF the throughput
# even though the kernel did the same work per resolved pair -- it passes --verify
# (which samples 50 pairs) and looks like a legitimate measurement. That is the Fase 6
# failure mode reappearing inside the B6 harness, so it gets pinned here rather than
# averaged over.
#
# This job prints resolved/abandoned for every repetition, so the anomalous rep can be
# identified instead of smoothed away.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
cd "$ROOT" || exit 1
source /kuhpc/sw/lmod/lmod/init/profile 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1

B=/beegfs/a474r867/b6diag
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. -Iinclude -DWFA_FLAT_BLOCK=128 \
    -DGENOALIGNER_BENCH_BACKEND='"rocm"' -o "$B" bench/tcus.cpp || exit 1

echo "=== host: $(hostname) ==="
echo "=== B6 diagnostic: resolved/abandoned per repetition ==="
echo

for K in default flat; do
  echo "######## kernel=$K  (len=1024 pairs=2000 ident=90 smax=256)"
  printf "%5s %10s %12s %10s %14s\n" "rep" "TCUPS" "resolved" "abandoned" "cells_res"
  for i in $(seq 1 7); do
    OUT=$("$B" --len 1024 --pairs 2000 --ident 90 --smax 256 --reps 20 --verify 50 --kernel $K 2>&1)
    RC=$?
    if [ "$RC" -ne 0 ]; then printf "%5d %10s %12s %10s %14s\n" "$i" "rc=$RC" "-" "-" "-"; continue; fi
    T=$(echo "$OUT" | grep -E 'kernel-only' | sed 's/.*: *//; s/ *TCUPS.*//')
    RA=$(echo "$OUT" | grep -E 'resolved /' | sed 's/.*: *//')
    RES=$(echo "$RA" | awk '{print $1}')
    ABD=$(echo "$RA" | awk '{print $3}')
    CR=$(echo "$OUT" | grep -E 'cells actually resolved' | sed 's/.*: *//; s/  (.*//')
    printf "%5d %10s %12s %10s %14s\n" "$i" "$T" "$RES" "$ABD" "$CR"
  done
  echo
done

echo "=== B6 diagnostic DONE ==="
echo "If the low-TCUP rep also shows fewer RESOLVED pairs, the metric is being"
echo "diluted by abandonment, not by the kernel being slower -- and the harness"
echo "must report resolved/abandoned alongside every throughput."
