#!/bin/bash
cd /beegfs/a474r867/genoaligner/repo
R=/kuhpc/sw/rocm/6.4.3
C=/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4
"$C/bin/nvcc" -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
    -gencode arch=compute_70,code=sm_70 \
    -I "$R/include" -I "$C/include" -I "$C/targets/x86_64-linux/include" \
    -I . -I include -lcuda -L "$C/lib64/stubs" -x cu \
    -o /tmp/genomsa_nv tools/genomsa.cpp src/msa/msa_gpu.cpp src/msa/msa_ref.cpp || exit 1
head -n 300 /beegfs/a474r867/phylogenyAI/data/genes_qc_pass/COI.fasta > /tmp/sub.fasta
FL="--psgp --gappy 0.95"
for r in 1 2 3; do
  /tmp/genomsa_nv /tmp/sub.fasta /tmp/g$r.aln $FL --tree-out /tmp/t$r.nwk 2>/dev/null
done
/tmp/genomsa_nv /tmp/sub.fasta /tmp/c.aln $FL --cpu --tree-out /tmp/tc.nwk 2>/dev/null
for r in 1 2 3; do cmp -s /tmp/t$r.nwk /tmp/tc.nwk && echo "tree$r == cpu" || echo "tree$r != cpu"; done
cmp -s /tmp/t1.nwk /tmp/t2.nwk && echo "t1==t2" || echo "t1!=t2"
cmp -s /tmp/t2.nwk /tmp/t3.nwk && echo "t2==t3" || echo "t2!=t3"
