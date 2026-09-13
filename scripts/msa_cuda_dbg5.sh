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
/tmp/genomsa_nv /tmp/sub.fasta /tmp/c.aln $FL --cpu 2>/tmp/cl.log
GENOMSA_LVL_DEBUG=1 /tmp/genomsa_nv /tmp/sub.fasta /tmp/g1.aln $FL 2>/tmp/l1.log
GENOMSA_LVL_DEBUG=1 /tmp/genomsa_nv /tmp/sub.fasta /tmp/g2.aln $FL 2>/tmp/l2.log
cmp -s /tmp/g1.aln /tmp/g2.aln && echo "g1==g2" || echo "g1!=g2"
cmp -s /tmp/g1.aln /tmp/c.aln && echo "g1==cpu" || echo "g1!=cpu"
echo "--- first log diff (run1 vs run2) ---"
diff /tmp/l1.log /tmp/l2.log | head -30
