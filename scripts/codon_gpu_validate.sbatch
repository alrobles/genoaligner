#!/bin/bash -l
#SBATCH --job-name=codon_gpu_val
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/codongpu_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/codongpu_%j.err

# On-device validation of the codon stage-2 phase-DP kernel
# (codon_refine_kernel, one thread per row).
#
#   1. build tools/genomsa with hipcc (gfx90a)
#   2. device-vs-CPU parity: --codon --local-frame on a subset through BOTH
#      paths -- byte-identical required (same DP body by construction)
#   3. full-gene GPU codon run + timing; compare vs --cpu refine walltime
#
# GENE/BIG overridable: sbatch --export=GENE=ADORA3,BIG=ND1 ...
set -o pipefail

PAI=/beegfs/a474r867/phylogenyAI
REPO=/beegfs/a474r867/genoaligner/repo
GENE=${GENE:-BMI1}
BIG=${BIG:-ND1}
SUBN=${SUBN:-150}
cd "$REPO" || exit 1

echo "=== host: $(hostname)  $(date) ==="
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -ffp-contract=off -std=c++17 -I. -Iinclude \
    -o /tmp/genomsa tools/genomsa.cpp src/msa/msa_gpu.cpp src/msa/msa_ref.cpp \
    src/msa/nj_gpu.cpp \
    || { echo "BUILD FAILED"; exit 1; }
echo "build ok"

FA="$PAI/data/genes_qc_pass/${GENE}.fasta"
FB="$PAI/data/genes_qc_pass/${BIG}.fasta"
GC=1; case "$BIG" in COI|CYTB|ND1|ND2) GC=2;; esac

RC=0
head -n $((SUBN * 2)) "$FA" > /tmp/sub_${GENE}.fasta

echo; echo "### parity subset: $GENE first $SUBN records (--codon --local-frame) ###"
/tmp/genomsa /tmp/sub_${GENE}.fasta /tmp/sub_gpu.codon \
    --codon --gc-def 1 --local-frame 2>/tmp/g1.log \
    || { echo "GPU run failed"; cat /tmp/g1.log; exit 1; }
tail -3 /tmp/g1.log
/tmp/genomsa /tmp/sub_${GENE}.fasta /tmp/sub_cpu.codon \
    --codon --gc-def 1 --local-frame --cpu 2>/tmp/c1.log \
    || { echo "CPU run failed"; cat /tmp/c1.log; exit 1; }
if diff -q /tmp/sub_gpu.codon /tmp/sub_cpu.codon >/dev/null; then
    echo "PARITY: PASS (device == CPU-ref, byte-identical)"
else
    echo "PARITY: FAIL -- device output differs"; RC=1
    diff /tmp/sub_gpu.codon /tmp/sub_cpu.codon | head -10
fi

echo; echo "### full $BIG (GPU codon, gc=$GC) ###"
{ time /tmp/genomsa "$FB" /tmp/${BIG}.gpu.codon \
    --codon --gc-def $GC --local-frame ; } 2>&1 | tail -5
echo; echo "### full $BIG (CPU codon, gc=$GC) ###"
{ time /tmp/genomsa "$FB" /tmp/${BIG}.cpu.codon \
    --codon --gc-def $GC --local-frame --cpu ; } 2>&1 | tail -5
if diff -q /tmp/${BIG}.gpu.codon /tmp/${BIG}.cpu.codon >/dev/null; then
    echo "PARITY-FULL: PASS"
else
    echo "PARITY-FULL: FAIL"; RC=1
fi

echo
[ "$RC" -eq 0 ] && echo "CODON-GPU-VAL: PASS" || echo "CODON-GPU-VAL: FAIL"
exit "$RC"
