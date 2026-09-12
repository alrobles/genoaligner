#!/bin/bash -l
#SBATCH --job-name=h7_flat
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h7flat_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h7flat_%j.err

# Fase 7 — the flat (fixed-block) score kernel on the MI210.
#
# Two separate purposes:
#
# 1. VERIFY THE THREAD REMAPPING. The CPU gate (bench/h7_flat_parity.cpp) runs at
#    blockDim=1, where the interior grid-stride loop degenerates to one thread per
#    slot -- it checks the layout but NOT the reassignment of diagonals to threads.
#    Only real threads can catch a remapping bug, and every regime is checked
#    against the CPU DP via the bench's --verify path (exit 5/6 on failure).
#
# 2. MEASURE, in the exact regimes where the ORIGINAL kernel's time scaled with
#    smax at constant real distance (job 29210725 gave 0.966/2.190/4.835/10.349 ms
#    for smax 32/64/128/256). If decoupling blockDim from smax works, these flatten.
#
# WFA_FLAT_BLOCK is swept because the right block size is a measurement, not a
# guess: the whole defect was a block size chosen for coverage instead of cost.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
OUT_DIR=/beegfs/a474r867/genoaligner/bench
mkdir -p "$OUT_DIR"

source /kuhpc/sw/lmod/9.3/init/bash 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1
cd "$ROOT" || exit 1

echo "=== host: $(hostname) ==="
echo "=== H7-flat: fixed-block score kernel (MI210) ==="
echo

rc_all=0

echo
echo "=============================================================="
echo "PART 0 — baseline: the ORIGINAL kernel, same regimes as before"
echo "=============================================================="
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. \
    -DGENOALIGNER_BENCH_BACKEND='"rocm"' -o /tmp/tcus_base bench/tcus.cpp
if [ $? -ne 0 ]; then echo ">>> baseline build failed"; exit 1; fi
for SM in 32 64 128 256; do
    echo
    echo "######## default, len=1024 pairs=2000 ident=98 smax=$SM"
    /tmp/tcus_base --len 1024 --pairs 2000 --ident 98 --smax $SM --reps 30 --kernel default \
        | grep -E 'kernel  :|launch  :|KERNEL \(mean|kernel-only|resolved /|OK:|>>>'
done

echo
echo "=============================================================="
echo "PART 1 — parity of the flat kernel under real threads"
echo "  a remapping bug shows up here as rc=6 (disagrees with CPU DP)"
echo "=============================================================="
for BLK in 64 128 256 512; do
    echo
    echo "######## parity, WFA_FLAT_BLOCK=$BLK"
    /kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. \
        -DWFA_FLAT_BLOCK=$BLK -DGENOALIGNER_BENCH_BACKEND='"rocm"' \
        -o /tmp/tcus_flat_$BLK bench/tcus.cpp
    if [ $? -ne 0 ]; then echo ">>> build failed for BLK=$BLK"; rc_all=1; continue; fi
    /tmp/tcus_flat_$BLK --len 1024 --pairs 500 --ident 98 --smax 128 --reps 5 \
        --verify 500 --kernel flat | tee -a "$OUT_DIR/h7_flat_${SLURM_JOB_ID}.txt"
    RC=${PIPESTATUS[0]}
    case "$RC" in
        0) : ;;
        5) echo ">>> UNVERIFIED (rc=5): nothing compared."; rc_all=1 ;;
        6) echo ">>> INVALID (rc=6): flat kernel disagrees with the CPU DP."; rc_all=1 ;;
        *) echo ">>> FAILED (rc=$RC)."; rc_all=1 ;;
    esac
done

echo
echo "=============================================================="
echo "PART 2 — smax sweep, flat kernel, block=128 vs the baseline"
echo "  baseline: 0.966 / 2.190 / 4.835 / 10.349 ms"
echo "=============================================================="
if [ -x /tmp/tcus_flat_128 ]; then
    for SM in 32 64 128 256; do
        echo
        echo "######## flat, len=1024 pairs=2000 ident=98 smax=$SM"
        /tmp/tcus_flat_128 --len 1024 --pairs 2000 --ident 98 --smax $SM --reps 30 --kernel flat \
            | grep -E 'kernel  :|WFA_FLAT_BLOCK|launch  :|occupancy|resident|KERNEL \(mean|kernel-only|resolved /|OK:|>>>'
    done
else
    echo ">>> block=128 build missing"; rc_all=1
fi

echo
echo "=== H7-flat RESULT ==="
if [ "$rc_all" -eq 0 ]; then echo "H7-flat: PASS"; else echo "H7-flat: FAIL"; fi
exit "$rc_all"
