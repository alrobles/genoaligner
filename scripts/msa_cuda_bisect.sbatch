#!/bin/bash -l
#SBATCH --job-name=msa_cuda_bisect
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/msabisect_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/msabisect_%j.err

# Bisect the V100 parity failure: which feature (psgp / gappy / both)
# makes the kernel diverge from the CPU reference.
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
echo "build ok"

FA="$PAI/data/genes_qc_pass/COI.fasta"
head -n 300 "$FA" > /tmp/sub_COI.fasta

for cfg in legacy psgp gappy both; do
  case $cfg in
    legacy) FL="--no-psgp --no-gappy";;
    psgp)   FL="--psgp --no-gappy";;
    gappy)  FL="--no-psgp --gappy 0.95";;
    both)   FL="--psgp --gappy 0.95";;
  esac
  /tmp/genomsa_nv /tmp/sub_COI.fasta /tmp/g_$cfg.aln $FL        2>/dev/null
  /tmp/genomsa_nv /tmp/sub_COI.fasta /tmp/c_$cfg.aln $FL --cpu  2>/dev/null
  if cmp -s /tmp/g_$cfg.aln /tmp/c_$cfg.aln; then
    echo "$cfg: PARITY PASS"
  else
    n=$(diff /tmp/g_$cfg.aln /tmp/c_$cfg.aln | grep -c '^<')
    echo "$cfg: PARITY FAIL ($n rows differ)"
  fi
done
