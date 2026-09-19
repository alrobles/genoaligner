#!/bin/bash -l
#SBATCH --job-name=h1c_container
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h1c_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h1c_%j.err

# H1-C TEST: the containerized build/run path, on an AMD MI210 node.
#
# Difference from h1_rocm.sbatch: NOTHING is loaded from Lmod. The compilers
# come from the .sif, and the GPU driver comes from the host via --rocm.
# If this passes, the whole class of "module load order / cmake PATH /
# GLIBCXX_3.4.32" failures is structurally eliminated -- not patched.
#
# Shebang must be '#!/bin/bash -l' (Slurm's non-interactive shell has no Lmod,
# and we still need the login environment for srun/mpirun conventions).
# No 'set -u': the site activation hooks read unset variables.

set -o pipefail

SIF=/beegfs/a474r867/genoaligner/images/genoaligner-compile.sif
ROOT=/beegfs/a474r867/genoaligner/repo

echo "=== host: $(hostname) ==="
echo "=== SIF:  $SIF ==="
if [ ! -f "$SIF" ]; then
    echo "H1C: FAIL -- container image not found at $SIF"
    exit 2
fi

# Report the host GPU independently of the container, so a container-side
# failure can be told apart from a driver-side failure.
echo "--- host GPU ---"
rocminfo 2>/dev/null | grep -E "gfx|Marketing Name" | head -4 \
    || echo "rocminfo unavailable on host"

echo
echo "--- container toolchain (no Lmod involved) ---"
apptainer run "$SIF" || { echo "H1C: FAIL (image would not run)"; exit 1; }

echo
echo "--- build genoaligner inside the container ---"
cd "$ROOT" || exit 1
apptainer exec --rocm \
    --bind /beegfs:/beegfs \
    --bind "$ROOT":"$ROOT" \
    --pwd "$ROOT" \
    "$SIF" \
    bash -c '
        set -e
        echo "hipcc: $(command -v hipcc)"
        rm -rf build/container
        cmake -S . -B build/container \
              -DCMAKE_CXX_COMPILER="$(command -v hipcc)" \
              -DGENOALIGNER_BACKEND=rocm \
              -DCMAKE_BUILD_TYPE=Release >/dev/null
        cmake --build build/container -j "$(nproc)"
        echo
        echo "--- run probe on MI210, from inside the container ---"
        ./build/container/genoaligner_probe
    '
RC=$?

echo
echo "=== H1C RESULT (containerized, AMD) ==="
if [ "$RC" -eq 0 ]; then
    echo "H1C_AMD: PASS"
else
    echo "H1C_AMD: FAIL (rc=$RC)"
fi
exit "$RC"
