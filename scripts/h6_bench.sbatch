#!/bin/bash -l
#SBATCH --job-name=h6_bench
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h6_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h6_%j.err

# Fase 6 first measurement: the phase split on the MI210, before any headline
# TCUPS is chosen. The point of this job is the SETUP SHARE, not the peak number:
# if host-side setup dominates end-to-end, then a kernel-only TCUPS is the only
# honest headline and the setup needs work separately; if it is small, both
# numbers can be reported without misleading anyone.
#
# Three regimes, because the split is a function of the work per pair:
#   short pairs (len=256, many)   -> setup-heavy end
#   mid   pairs (len=1024)        -> the project's phylogenetic regime
#   long  pairs (len=4096, fewer) -> kernel-heavy end
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
OUT_DIR=/beegfs/a474r867/genoaligner/bench
mkdir -p "$OUT_DIR"
BIN=/tmp/tcus_rocm

echo "=== host: $(hostname) ==="
echo "=== H6: Fase 6 phase-separated timing probe (MI210) ==="
echo

source /kuhpc/sw/lmod/9.3/init/bash 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1
hipcc --version 2>/dev/null | head -1 || /kuhpc/sw/rocm/6.4.3/bin/hipcc --version | head -1

cd "$ROOT" || exit 1

echo
echo "--- build ---"
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. \
    -DGENOALIGNER_BENCH_BACKEND='"rocm"' \
    -o "$BIN" bench/tcus.cpp
RC_BUILD=$?
echo "BUILD_RC=$RC_BUILD"
[ "$RC_BUILD" -ne 0 ] && { echo "H6: BUILD FAILED"; exit 1; }

rc_all=0
run() {
    local label="$1"; shift
    echo
    echo "=============================================================="
    echo "REGIME: $label"
    echo "=============================================================="
    "$BIN" "$@" | tee -a "$OUT_DIR/h6_rocm_${SLURM_JOB_ID}.txt"
    local rc=${PIPESTATUS[0]}
    [ "$rc" -ne 0 ] && rc_all=1
    return 0
}

# ident 90% is the regime WFA is built for; 70% stresses a larger s and is the
# case where the wavefront actually has to work rather than skim.
#
# smax IS A REGIME PARAMETER, NOT A CONSTANT. At ident=90% over len=1024 the true
# distance is ~100 substitutions, so smax=64 ABANDONS most pairs and the kernel
# returns early -- measuring the guard, not the algorithm. The variants below set
# smax above the expected distance so the pairs are actually resolved; the probe
# now prints resolved/abandoned and refuses to present a TCUPS over abandoned-only
# pairs.
run "len=256,  smax=64,  8000 pairs, ident=90%"  --len 256  --pairs 8000 --ident 90 --smax 64  --reps 20
run "len=1024, smax=64,  2000 pairs, ident=90%"  --len 1024 --pairs 2000 --ident 90 --smax 64  --reps 20
run "len=1024, smax=256, 2000 pairs, ident=90%"  --len 1024 --pairs 2000 --ident 90 --smax 256 --reps 20
run "len=1024, smax=256, 2000 pairs, ident=70%"  --len 1024 --pairs 2000 --ident 70 --smax 256 --reps 20
run "len=4096, smax=1024, 500 pairs, ident=99%"  --len 4096 --pairs 500  --ident 99 --smax 1024 --reps 15

echo
echo "=== H6 RESULT ==="
if [ "$rc_all" -eq 0 ]; then echo "H6: PASS"; else echo "H6: FAIL"; fi
exit "$rc_all"
