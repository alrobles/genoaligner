#!/bin/bash -l
#SBATCH --job-name=msa_cuda_dbg
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/msadbg_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/msadbg_%j.err
set -o pipefail
PAI=/beegfs/a474r867/phylogenyAI
REPO=/beegfs/a474r867/genoaligner/repo
R=/kuhpc/sw/rocm/6.4.3
C=/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4
cd "$REPO" || exit 1
"$C/bin/nvcc" -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
    -gencode arch=compute_70,code=sm_70 \
    -I "$R/include" -I "$C/include" -I "$C/targets/x86_64-linux/include" \
    -I . -I include -lcuda -L "$C/lib64/stubs" -x cu \
    -o /tmp/genomsa_nv tools/genomsa.cpp src/msa/msa_gpu.cpp src/msa/msa_ref.cpp \
    || { echo "BUILD FAILED"; exit 1; }
FA="$PAI/data/genes_qc_pass/COI.fasta"
head -n 300 "$FA" > /tmp/sub.fasta
FL="--psgp --gappy 0.95"
/tmp/genomsa_nv /tmp/sub.fasta /tmp/c.aln $FL --cpu 2>/dev/null
for K in wf serial; do
  GENOMSA_MSA_KERNEL=$K /tmp/genomsa_nv /tmp/sub.fasta /tmp/g_$K.aln $FL 2>/dev/null
  if cmp -s /tmp/g_$K.aln /tmp/c.aln; then echo "$K: PARITY PASS";
  else echo "$K: FAIL ($(diff /tmp/g_$K.aln /tmp/c.aln | grep -c '^<') rows)"; fi
done
# determinism: same kernel twice
GENOMSA_MSA_KERNEL=wf /tmp/genomsa_nv /tmp/sub.fasta /tmp/g_wf2.aln $FL 2>/dev/null
cmp -s /tmp/g_wf.aln /tmp/g_wf2.aln && echo "wf rerun: identical" || echo "wf rerun: NONDETERMINISTIC"
