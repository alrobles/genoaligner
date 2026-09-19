#!/bin/bash -l
#SBATCH --job-name=h8_b6d2
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h8b6d2_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h8b6d2_%j.err

# B6 follow-up — the len=2048 smax=128 ident=95% regime is UNSTABLE on MI210.
#
# Job 29226478 raw reps for that regime (default TCUPS):
#   1.209  0.320  1.209  1.209  0.915  0.558  1.209
# Not a smooth spread: DISCRETE modes, with 1.209 recurring. Same node, same job,
# same configuration -- so this is not node drift, it is something that changes
# between launches of the same kernel.
#
# The other MI210 regimes in the same job were perfectly stable (0.472 x7, spread
# 0.0%), which rules out "this node is noisy" as the explanation. So the instability
# is specific to THIS regime, and the candidate is that its true distance sits close
# to smax, making the number of wavefronts actually walked vary.
#
# This job prints, for that regime only, the quantities that decide it:
#   distance mean/max, wavefronts walked/offered, resolved/abandoned, and the timing,
# across repeated runs -- so the mechanism is identified rather than averaged over.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
cd "$ROOT" || exit 1
source /kuhpc/sw/lmod/lmod/init/profile 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1

B=/beegfs/a474r867/b6diag2
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. -Iinclude -DWFA_FLAT_BLOCK=128 \
    -DGENOALIGNER_BENCH_BACKEND='"rocm"' -o "$B" bench/tcus.cpp || exit 1

echo "=== host: $(hostname) ==="
echo "=== B6 follow-up: why is len=2048 smax=128 ident=95% unstable? ==="
echo

for K in default flat; do
  echo "######## kernel=$K"
  printf "%4s %9s %9s %9s %8s %9s %9s %12s\n" \
         "rep" "TCUPS" "dist_mean" "dist_max" "wf_off" "wf_used" "resolved" "abandoned"
  for i in $(seq 1 7); do
    F=/beegfs/a474r867/genoaligner/logs/b6d2_rep.txt
    "$B" --len 2048 --pairs 300 --ident 95 --smax 128 --reps 20 --verify 50 --kernel $K > "$F" 2>&1
    [ $? -ne 0 ] && { printf "%4d  rc!=0\n" "$i"; continue; }
    T=$(grep -E '^  kernel-only' "$F" | sed 's/.*: *//; s/ *TCUPS.*//')
    DM=$(grep -E '^  distance \(score\)' "$F" | sed -n 's/.*: *\([0-9.]*\) \/.*/\1/p')
    DX=$(grep -E '^  distance \(score\)' "$F" | sed -n 's/.*\/ *\([0-9]*\).*/\1/p')
    WF=$(grep -E '^  wavefronts walked' "$F" | sed -n 's/.*: *\([0-9.]*\) \/ *\([0-9]*\).*/\2/p')
    WU=$(grep -E '^  wavefronts walked' "$F" | sed -n 's/.*: *\([0-9.]*\) \/.*/\1/p')
    RS=$(grep -E '^  resolved /' "$F" | sed -n 's/^  resolved \/ abandoned *: *\([0-9]*\) \/.*/\1/p')
    AB=$(grep -E '^  resolved /' "$F" | sed -n 's/^  resolved \/ abandoned *: *[0-9]* \/ *\([0-9]*\).*/\1/p')
    printf "%4d %9s %9s %9s %8s %9s %9s %12s\n" "$i" "$T" "$DM" "$DX" "$WF" "$WU" "$RS" "$AB"
  done
  echo
done

echo "=== B6 follow-up DONE ==="
echo "If wf_used or resolved varies across reps, the kernel is doing DIFFERENT WORK"
echo "in identical runs and the throughput spread is that, not noise."
