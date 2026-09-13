#!/bin/bash -l
#SBATCH --job-name=h12_sw_parasail
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h12par_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h12par_%j.err

# H12: SW throughput -- our GPU score kernel vs parasail (CPU, SIMD).
#
# Fase D, first measurement. The design is B6's: one job, 7 interleaved reps
# (gpu-dev / gpu-wall / parasail-1thread / parasail-8threads inside EVERY rep),
# mean+-sd+min+max+spread, and every rep re-checks all scores against the
# independent serial reference -- a fast wrong answer fails the run.
#
# TARGET: an idle node (submitted with --nodelist to an idle MI210 node) --
# this cluster time-slices GPUs by shard, so a shared node would contaminate
# every column; see docs/RESULTADO_H8_SW1_WARP.md §3.
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
PS=/beegfs/a474r867/genoaligner/parasail
B=/beegfs/a474r867/genoaligner/gate_build_h12/sw_parasail
mkdir -p "$(dirname "$B")"
cd "$ROOT" || exit 1

echo "=== host: $(hostname) ==="
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
lscpu | grep -E "^Model name|^CPU\(s\)|Flags" | head -3 || true
rocm-smi --showpids 2>/dev/null || true
echo

# parasail was built with the system gcc; its .a is plain host code. Two-stage
# build because hipcc's driver passes .a files through the GPU compile pass as
# if they were sources (they are not valid UTF-8, the job fails there) --
# compile the TU first, then link.
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -pthread \
    -I. -Iinclude -I"$PS" \
    -c bench/sw_parasail.cpp -o "$B.o" || { echo "BUILD FAILED (compile)"; exit 1; }
/kuhpc/sw/rocm/6.4.3/bin/hipcc -pthread \
    -o "$B" "$B.o" "$PS/build/libparasail.a" \
    || { echo "BUILD FAILED (link)"; exit 1; }

echo "=== run ==="
"$B" || { echo "=== H12: FAIL ==="; exit 1; }
echo
echo "=== H12: PASS ==="
