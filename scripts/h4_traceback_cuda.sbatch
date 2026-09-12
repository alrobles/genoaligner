#!/bin/bash -l
#SBATCH --job-name=h4_trace_cuda
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:q6000:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h4cuda_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h4cuda_%j.err

# H4-CUDA: the SAME Fase 4 traceback source built with nvcc and run on an NVIDIA
# GPU. Companion to h4_traceback.sbatch (MI210).
#
# The point is the masterplan's "one source, two backends" claim, applied to the
# trace kernel and not just the score kernel. The mechanism is ROCm's header
# layer (hip/nvidia_detail) -- no shim, no hipify, no container. See the
# ku-hpc-gpu-toolchain skill, §10.
#
# CUDA 12.4, NOT 13.0: nvidia_detail targets CUDA 12.x.
set -o pipefail

R=/kuhpc/sw/rocm/6.4.3
C=/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4
ROOT=/beegfs/a474r867/genoaligner/repo
N_CASES=${N_CASES:-207}
# Emit to beegfs, NOT /tmp. See h4_traceback.sbatch for why: the /tmp path is
# node-local and vanishes with the job, which is how the Fase 4 external gate
# ended up with nothing to check.
OUT_DIR=/beegfs/a474r867/genoaligner/out
mkdir -p "$OUT_DIR"
CIGAR_TSV=$OUT_DIR/cigars_gpu_cuda_${SLURM_JOB_ID}.tsv

echo "=== host: $(hostname) ==="
echo "=== H4-CUDA: Fase 4 traceback on NVIDIA (nvcc, same source) ==="
echo

nvidia-smi -L 2>/dev/null || echo "no nvidia-smi"

echo "--- nvcc ---"
"$C/bin/nvcc" --version | tail -2

cd "$ROOT" || exit 1

# sm_75 = Turing (q6000). Same flags as the H2-CUDA job, so a difference in the
# outcome is a difference in the code, not the build.
#
# ONE FLAG THE H2-CUDA JOB DID NOT NEED: -lcuda. The trace kernel calls
# hipDeviceGetAttribute, which the nvidia_detail layer maps to cuDeviceGetAttribute
# -- a DRIVER-API symbol. The score kernel never queried a device attribute, so it
# linked without the driver; the traceback cannot. Measured on the login node:
# without -lcuda the link fails with "undefined reference to cuDeviceGetAttribute",
# and the stubs dir is enough to resolve it at build time (the real libcuda comes
# from the host driver at run time).
echo
echo "--- build harness (nvcc, __HIP_PLATFORM_NVIDIA__) ---"
"$C/bin/nvcc" -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
    -gencode arch=compute_75,code=sm_75 \
    -I "$R/include" \
    -I "$C/include" \
    -I "$C/targets/x86_64-linux/include" \
    -I . \
    -lcuda -L "$C/lib64/stubs" \
    -x cu -o /tmp/wfa_parity_trace_cuda \
    tests/parity/wfa_parity.cpp
RC_BUILD=$?
echo "BUILD_RC=$RC_BUILD"
if [ "$RC_BUILD" -ne 0 ]; then
    echo "H4-CUDA: BUILD FAILED"
    exit 1
fi

echo
echo "--- embedded GPU code ---"
"$C/bin/cuobjdump" --list-elf /tmp/wfa_parity_trace_cuda 2>/dev/null | head -4

echo
echo "--- run: traceback ---"
/tmp/wfa_parity_trace_cuda "$N_CASES" --traceback --emit "$CIGAR_TSV"
RC=$?

echo
echo "--- stage 3: CIGAR score vs edlib (external implementation) on GPU output ---"
python3 tests/parity/check_cigar.py "$CIGAR_TSV"
RC_EDLIB=$?
if [ "$RC_EDLIB" -ne 0 ]; then RC=1; fi

echo
echo "=== H4-CUDA RESULT ==="
if [ "$RC" -eq 0 ]; then echo "H4-CUDA: PASS"; else echo "H4-CUDA: FAIL (rc=$RC)"; fi
exit "$RC"
