#!/bin/bash -l
#SBATCH --job-name=msa_api_cuda
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/msaapicuda_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/msaapicuda_%j.err

# Public MSA API NVIDIA validation (docs/VALIDATION_MSA_API_CLUSTER.md, V2):
# same .hip sources through nvcc + ROCm's nvidia_detail layer. Proves the
# "portable multiarch" claim at the PUBLIC API level, not just the engine.
#
# GRES/ARCH must agree (the script checks nvcc, not the pairing):
#   v100 -> --gres=gpu:v100:1  GENOALIGNER_ARCH=70   (this file's defaults)
#   a100 -> --gres=gpu:a100:1  GENOALIGNER_ARCH=80
#   q6000 -> --gres=gpu:q6000:1 GENOALIGNER_ARCH=75
set -o pipefail

PAI=/beegfs/a474r867/phylogenyAI
REPO=${REPO:-/beegfs/a474r867/genoaligner/repo-msa-api}
GENE=${GENE:-BMI1}
GENE2=${GENE2:-CYTB}
SUBN=${SUBN:-150}
export GENOALIGNER_ARCH=${GENOALIGNER_ARCH:-70}
cd "$REPO" || exit 1

echo "=== host: $(hostname)  $(date) ==="
echo "=== repo: $REPO  $(git log --oneline -1 2>/dev/null) ==="
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi"

RC=0

# ---- 1. cmake build (nvcc + nvidia_detail, sm_$GENOALIGNER_ARCH) -----------
echo "### cmake build (scripts/build_cuda.sh, sm_${GENOALIGNER_ARCH}) ###"
bash scripts/build_cuda.sh || { echo "BUILD FAILED"; exit 1; }
B=build/cuda

# ---- 2. ctest -------------------------------------------------------------
echo; echo "### ctest (all -- device tests run, not skipped) ###"
ctest --test-dir "$B" --output-on-failure || RC=1

# ---- 3. the new device test, directly -------------------------------------
echo; echo "### genoaligner_msa_api_device_test ###"
"$B/genoaligner_msa_api_device_test" || RC=1

# ---- 4. public-API parity on real data ------------------------------------
FA="$PAI/data/genes_qc_pass/${GENE}.fasta"
FC="$PAI/data/genes_qc_pass/${GENE2}.fasta"
for f in "$FA" "$FC"; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done

head -n $((SUBN * 2)) "$FA" > /tmp/sub_${GENE}.fasta
echo; echo "### public API parity: $GENE first $SUBN records (dna, CUDA) ###"
"$B/genoaligner_example_msa" /tmp/sub_${GENE}.fasta          > /tmp/sub_host.aln
"$B/genoaligner_example_msa" /tmp/sub_${GENE}.fasta --device > /tmp/sub_dev.aln
head -1 /tmp/sub_host.aln; head -1 /tmp/sub_dev.aln
if diff -q <(tail -n +3 /tmp/sub_host.aln) <(tail -n +3 /tmp/sub_dev.aln) >/dev/null; then
    echo "PARITY: PASS (device == host, byte-identical, on CUDA device)"
else
    echo "PARITY: FAIL"; RC=1
    diff <(tail -n +3 /tmp/sub_host.aln) <(tail -n +3 /tmp/sub_dev.aln) | head -10
fi

head -n $((SUBN * 2)) "$FC" > /tmp/sub_${GENE2}.fasta
echo; echo "### public API parity: $GENE2 (--codon --gc 2, CUDA) ###"
"$B/genoaligner_example_msa" /tmp/sub_${GENE2}.fasta --codon --gc 2          > /tmp/c_host.aln
"$B/genoaligner_example_msa" /tmp/sub_${GENE2}.fasta --codon --gc 2 --device > /tmp/c_dev.aln
if diff -q <(tail -n +3 /tmp/c_host.aln) <(tail -n +3 /tmp/c_dev.aln) >/dev/null; then
    echo "CODON-PARITY: PASS"
else
    echo "CODON-PARITY: FAIL"; RC=1
fi

echo
[ "$RC" -eq 0 ] && echo "MSA-API-CUDA-VAL: PASS" || echo "MSA-API-CUDA-VAL: FAIL"
exit "$RC"
