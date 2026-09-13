#!/bin/bash -l
#SBATCH --job-name=msa_cuda_dbg2
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/msadbg2_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/msadbg2_%j.err
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
# host determinism
/tmp/genomsa_nv /tmp/sub.fasta /tmp/c1.aln $FL --cpu 2>/dev/null
/tmp/genomsa_nv /tmp/sub.fasta /tmp/c2.aln $FL --cpu 2>/dev/null
cmp -s /tmp/c1.aln /tmp/c2.aln && echo "cpu rerun: identical" || echo "cpu rerun: NONDETERMINISTIC"
# GPU with single host thread (serial dist+NJ)
GENOMSA_THREADS=1 /tmp/genomsa_nv /tmp/sub.fasta /tmp/g_t1.aln $FL 2>/dev/null
GENOMSA_THREADS=1 /tmp/genomsa_nv /tmp/sub.fasta /tmp/g_t1b.aln $FL 2>/dev/null
cmp -s /tmp/g_t1.aln /tmp/g_t1b.aln && echo "gpu T1 rerun: identical" || echo "gpu T1 rerun: NONDETERMINISTIC"
cmp -s /tmp/g_t1.aln /tmp/c1.aln && echo "gpu T1 vs cpu: PASS" || echo "gpu T1 vs cpu: FAIL ($(diff /tmp/g_t1.aln /tmp/c1.aln | grep -c '^<') rows)"
# serial kernel determinism (3 runs)
for k in 1 2 3; do GENOMSA_MSA_KERNEL=serial /tmp/genomsa_nv /tmp/sub.fasta /tmp/s$k.aln $FL 2>/dev/null; done
cmp -s /tmp/s1.aln /tmp/s2.aln && cmp -s /tmp/s2.aln /tmp/s3.aln && echo "serial x3: identical" || echo "serial x3: NONDETERMINISTIC"
cmp -s /tmp/s1.aln /tmp/c1.aln && echo "serial vs cpu: PASS" || echo "serial vs cpu: FAIL"
