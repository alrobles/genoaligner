#!/bin/bash -l
#SBATCH --job-name=h2_wfa_parity
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h2_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h2_%j.err

# H2 TEST: WFA score parity, GPU vs independent CPU reference.
#
# Fusion criterion: >=95% agreement on ~1000 pairs. Below that the formulation
# is wrong and optimisation would be premature.
#
# Builds inside the container (no Lmod), runs on the MI210 via --rocm.
# The reference is a classic O(nm) DP -- a DIFFERENT algorithm from the
# wavefront, so agreement is real evidence and not a tautology.

set -o pipefail

SIF=/beegfs/a474r867/genoaligner/images/genoaligner-compile.sif
ROOT=/beegfs/a474r867/genoaligner/repo
N_CASES=${N_CASES:-1000}

echo "=== host: $(hostname) ==="
echo "=== H2: WFA score parity on MI210 ==="

cd "$ROOT" || exit 1

apptainer exec --rocm \
    --bind /beegfs:/beegfs \
    --bind "$ROOT":"$ROOT" \
    --pwd "$ROOT" \
    "$SIF" \
    bash -c "
        set -e
        echo '--- compiler ---'
        hipcc --version | head -1
        echo
        echo '--- build harness ---'
        hipcc -O2 -std=c++17 -I. \
              -o /tmp/wfa_parity \
              tests/parity/wfa_parity.cpp
        echo
        echo '--- run ---'
        /tmp/wfa_parity $N_CASES
    "
RC=$?

echo
echo "=== H2 RESULT ==="
if [ "$RC" -eq 0 ]; then echo "H2: PASS"; else echo "H2: FAIL (rc=$RC)"; fi
exit "$RC"
