#!/bin/bash -l
#SBATCH --job-name=h8_sw2_ab
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h8sw2ab_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h8sw2ab_%j.err

# SW speedup: sequential (blockDim=1) vs warp-per-row, interleaved in one job.
# B6 design: same job, 7 reps, A then B inside each rep, device time via
# hipEvent. Submit to an IDLE node (--nodelist) so no co-tenant can inject
# stalls into the measurement -- this cluster shares GPUs via shards.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
LOGD=/beegfs/a474r867/genoaligner/logs
cd "$ROOT" || exit 1
source /kuhpc/sw/lmod/lmod/init/profile 2>/dev/null
ml purge >/dev/null 2>&1
ml load rocm/6.4.3 >/dev/null 2>&1

B=/beegfs/a474r867/sw_ab
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. -Iinclude \
    -o "$B" bench/sw_ab.cpp || { echo "BUILD FAILED"; exit 1; }

echo "=== host: $(hostname) ==="
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
/kuhpc/sw/rocm/6.4.3/bin/rocm-smi --showpids 2>/dev/null || true
echo "=== SW A/B: sequential vs warp-per-row ==="
echo
"$B" | tee "$LOGD/sw_ab_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
echo
echo "=== H8-SW2 RESULT ==="
if [ "$RC" -eq 0 ]; then echo "H8-SW2: PASS";
elif [ "$RC" -eq 2 ]; then echo "H8-SW2: NOISY (co-tenant suspected)";
else echo "H8-SW2: FAIL (rc=$RC)"; fi
exit "$RC"
