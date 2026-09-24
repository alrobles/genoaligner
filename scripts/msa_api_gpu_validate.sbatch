#!/bin/bash -l
#SBATCH --job-name=msa_api_val
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/msaapival_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/msaapival_%j.err

# Public MSA API on-device validation (docs/VALIDATION_MSA_API_CLUSTER.md, V1).
#
# The engine driver was already validated (msa_gpu_validate.sbatch); what is
# NEW in devin/msa-public-api is the cmake build of the public entries:
#   libgenoaligner (api_msa_gpu.cpp) -> genomsa_gpu (msa_gpu.cpp + kernels)
#
#   1. cmake build via scripts/build_rocm.sh on the node (gfx90a auto)
#   2. full ctest -- msa_device_runs runs for real, not skipped
#   3. genoaligner_msa_api_device_test: must PASS, not exit 77
#   4. public-API parity on real data: genoaligner_example_msa host vs
#      --device -- aligned body byte-identical (summary line differs by
#      design: it reports engine=host|device)
#   5. codon path on a real mt CDS ($GENE2, gc=2)
#   6. host-vs-device walltime on $BIG -- the speedup number for the paper
#   7. full cpu gate with HIPCC set: stage 6 (real sequences) runs on device
#
# Overridable: sbatch --export=GENE=BMI1,GENE2=CYTB,BIG=ND1,SUBN=150 ...
set -o pipefail

PAI=/beegfs/a474r867/phylogenyAI
REPO=${REPO:-/beegfs/a474r867/genoaligner/repo-msa-api}
GENE=${GENE:-BMI1}
GENE2=${GENE2:-CYTB}
BIG=${BIG:-ND1}
SUBN=${SUBN:-150}
cd "$REPO" || exit 1

echo "=== host: $(hostname)  $(date) ==="
echo "=== repo: $REPO  $(git log --oneline -1 2>/dev/null) ==="

RC=0

# ---- 1. cmake build (hipcc, gfx90a detected on the node) ------------------
ml load rocm/6.4.3 2>/dev/null || export PATH=/kuhpc/sw/rocm/6.4.3/bin:$PATH
echo "### cmake build (scripts/build_rocm.sh) ###"
bash scripts/build_rocm.sh || { echo "BUILD FAILED"; exit 1; }
B=build/rocm

# ---- 2. ctest: every test runs for real on this node ----------------------
echo; echo "### ctest (all -- device tests run, not skipped) ###"
ctest --test-dir "$B" --output-on-failure || RC=1

# ---- 3. the new device test, directly -------------------------------------
echo; echo "### genoaligner_msa_api_device_test ###"
"$B/genoaligner_msa_api_device_test" || RC=1

# ---- 4. public-API parity on real data (byte-identical body) --------------
FA="$PAI/data/genes_qc_pass/${GENE}.fasta"
FB="$PAI/data/genes_qc_pass/${BIG}.fasta"
FC="$PAI/data/genes_qc_pass/${GENE2}.fasta"
for f in "$FA" "$FB" "$FC"; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done

head -n $((SUBN * 2)) "$FA" > /tmp/sub_${GENE}.fasta
echo; echo "### public API parity: $GENE first $SUBN records (dna) ###"
"$B/genoaligner_example_msa" /tmp/sub_${GENE}.fasta          > /tmp/sub_host.aln
"$B/genoaligner_example_msa" /tmp/sub_${GENE}.fasta --device > /tmp/sub_dev.aln
head -1 /tmp/sub_host.aln; head -1 /tmp/sub_dev.aln
if diff -q <(tail -n +3 /tmp/sub_host.aln) <(tail -n +3 /tmp/sub_dev.aln) >/dev/null; then
    echo "PARITY: PASS (device == host, byte-identical, on-device)"
else
    echo "PARITY: FAIL -- device output differs"; RC=1
    diff <(tail -n +3 /tmp/sub_host.aln) <(tail -n +3 /tmp/sub_dev.aln) | head -10
fi

# ---- 5. codon path on a real mitochondrial CDS ----------------------------
head -n $((SUBN * 2)) "$FC" > /tmp/sub_${GENE2}.fasta
echo; echo "### public API parity: $GENE2 first $SUBN records (--codon --gc 2) ###"
"$B/genoaligner_example_msa" /tmp/sub_${GENE2}.fasta --codon --gc 2          > /tmp/c_host.aln
"$B/genoaligner_example_msa" /tmp/sub_${GENE2}.fasta --codon --gc 2 --device > /tmp/c_dev.aln
head -1 /tmp/c_host.aln; head -1 /tmp/c_dev.aln
head -3 /tmp/c_dev.aln | tail -1   # shows real QC in the record header
if diff -q <(tail -n +3 /tmp/c_host.aln) <(tail -n +3 /tmp/c_dev.aln) >/dev/null; then
    echo "CODON-PARITY: PASS"
else
    echo "CODON-PARITY: FAIL"; RC=1
    diff <(tail -n +3 /tmp/c_host.aln) <(tail -n +3 /tmp/c_dev.aln) | head -10
fi

# ---- 6. timing: host vs device on the big gene ----------------------------
echo; echo "### timing: $BIG full gene ###"
echo "-- host --"
time "$B/genoaligner_example_msa" "$FB"          > /tmp/big_host.aln
echo "-- device --"
time "$B/genoaligner_example_msa" "$FB" --device > /tmp/big_dev.aln
if diff -q <(tail -n +3 /tmp/big_host.aln) <(tail -n +3 /tmp/big_dev.aln) >/dev/null; then
    echo "BIG-PARITY: PASS ($BIG identical)"
else
    echo "BIG-PARITY: FAIL"; RC=1
fi

# ---- 7. full gate with HIPCC: stage 6 runs on the real device -------------
echo; echo "### cpu gate with HIPCC (stage 6 runs for real) ###"
HIPCC="$(command -v hipcc)" bash scripts/check_kernel_cpu.sh 2>&1 | tail -25 \
    || RC=1

echo
[ "$RC" -eq 0 ] && echo "MSA-API-VAL: PASS" || echo "MSA-API-VAL: FAIL"
exit "$RC"
