#!/bin/bash -l
#SBATCH --job-name=msa_cuda_val
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/msaval_cuda_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/msaval_cuda_%j.err

# genomsa NVIDIA validation: same .hip sources through nvcc + ROCm's
# nvidia_detail layer (see scripts/build_cuda.sh). Parity vs CPU reference
# must be byte-identical -- same discipline as the MI210 run.
set -o pipefail

PAI=/beegfs/a474r867/phylogenyAI
REPO=/beegfs/a474r867/genoaligner/repo
OUT=$PAI/data/msa_genomsa
R=/kuhpc/sw/rocm/6.4.3
C=/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4
GENE=${GENE:-COI}
mkdir -p "$OUT"
cd "$REPO" || exit 1

echo "=== host: $(hostname)  $(date) ==="
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi"
"$C/bin/nvcc" --version | tail -2

"$C/bin/nvcc" -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
    -gencode arch=compute_70,code=sm_70 \
    -I "$R/include" -I "$C/include" -I "$C/targets/x86_64-linux/include" \
    -I . -I include -lcuda -L "$C/lib64/stubs" -x cu \
    -o /tmp/genomsa_nv tools/genomsa.cpp src/msa/msa_gpu.cpp src/msa/msa_ref.cpp \
    || { echo "BUILD FAILED"; exit 1; }
echo "build ok"

FA="$PAI/data/genes_qc_pass/${GENE}.fasta"
[ -s "$FA" ] || { echo "missing $FA"; exit 1; }
RC=0

head -n 300 "$FA" > /tmp/sub_${GENE}.fasta
echo; echo "### parity subset: $GENE first 150 records (V100) ###"
/tmp/genomsa_nv /tmp/sub_${GENE}.fasta /tmp/sub_gpu.aln 2>/tmp/g.log || { echo "GPU run failed"; cat /tmp/g.log; exit 1; }
cat /tmp/g.log
/tmp/genomsa_nv /tmp/sub_${GENE}.fasta /tmp/sub_cpu.aln --cpu 2>/tmp/c.log || { echo "CPU run failed"; cat /tmp/c.log; exit 1; }
if diff -q /tmp/sub_gpu.aln /tmp/sub_cpu.aln >/dev/null; then
    echo "PARITY: PASS (nvcc/V100 == CPU-ref, byte-identical)"
else
    echo "PARITY: FAIL"; RC=1; diff /tmp/sub_gpu.aln /tmp/sub_cpu.aln | head -10
fi

echo; echo "### full $GENE (V100, wavefront) ###"
time /tmp/genomsa_nv "$FA" "$OUT/${GENE}.genomsa.aln.nv.fasta" 2>&1 | tail -4
if [ -f "$OUT/${GENE}.genomsa.aln.fasta" ]; then
    if diff -q "$OUT/${GENE}.genomsa.aln.nv.fasta" "$OUT/${GENE}.genomsa.aln.fasta" >/dev/null; then
        echo "X-VENDOR: PASS (V100 == MI210, byte-identical)"
    else
        echo "X-VENDOR: DIFFERENT (expected to differ only if float op order differs)"
    fi
fi

echo; [ "$RC" -eq 0 ] && echo "MSA-VAL-NV: PASS" || echo "MSA-VAL-NV: FAIL"
exit "$RC"
