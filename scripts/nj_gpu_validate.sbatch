#!/bin/bash -l
#SBATCH --job-name=nj_gpu0_val
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/njgpu_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/njgpu_%j.err

# NJ0 on-device validation (docs/SPEC_NJ_GPU.md).
#
#   1. build tests/msa/test_nj_gpu.cpp and tools/nj_bench.cpp with hipcc
#   2. synthetic parity (ties, degenerate n, additive metrics) on the device
#   3. every gene in $GENES_DIR: nj_tree_gpu vs nj_tree_mt (16 threads) --
#      Tree.nodes must be IDENTICAL; timing breakdown per gene to $OUT/nj_gpu0_<job>.tsv
#   4. end-to-end genomsa on $GENE with the GPU guide tree vs GENOMSA_NJ=cpu:
#      alignments must be byte-identical
#
# Overrides: sbatch --export=REPO=...,GENES_DIR=...,GENE=COI scripts/nj_gpu_validate.sbatch
set -o pipefail

PAI=/beegfs/a474r867/phylogenyAI
REPO=${REPO:-/beegfs/a474r867/genoaligner/repo}
GENES_DIR=${GENES_DIR:-$PAI/data/genes_qc_pass}
OUT=${OUT_DIR:-$PAI/data/nj_gpu0}
GENE=${GENE:-COI}
HIPCC=${HIPCC:-/kuhpc/sw/rocm/6.4.3/bin/hipcc}
T=${SLURM_CPUS_PER_TASK:-16}
mkdir -p "$OUT"
cd "$REPO" || exit 1

echo "=== host: $(hostname)  $(date)  job ${SLURM_JOB_ID:-local}  threads $T ==="
echo "=== commit: $(git rev-parse --short HEAD 2>/dev/null)  branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null) ==="
rocm-smi --showproductname 2>/dev/null | grep -i "card series" | head -1

BIN=/tmp/njgpu_${SLURM_JOB_ID:-local}
mkdir -p "$BIN"
"$HIPCC" -O2 -std=c++17 -I. -Iinclude -o "$BIN/nj_gpu_test" \
    tests/msa/test_nj_gpu.cpp src/msa/nj_gpu.cpp src/msa/msa_ref.cpp \
    || { echo "BUILD FAILED (test_nj_gpu)"; exit 1; }
"$HIPCC" -O2 -std=c++17 -I. -Iinclude -o "$BIN/nj_bench" \
    tools/nj_bench.cpp src/msa/nj_gpu.cpp src/msa/msa_ref.cpp \
    || { echo "BUILD FAILED (nj_bench)"; exit 1; }
"$HIPCC" -O2 -std=c++17 -I. -Iinclude -o "$BIN/genomsa" \
    tools/genomsa.cpp src/msa/msa_gpu.cpp src/msa/nj_gpu.cpp src/msa/msa_ref.cpp \
    || { echo "BUILD FAILED (genomsa)"; exit 1; }
echo "build ok"

RC=0

# ---- 2. synthetic parity on the device ----------------------------------
echo; echo "### synthetic parity (test_nj_gpu) ###"
"$BIN/nj_gpu_test" | tail -6
R=${PIPESTATUS[0]}
if [ "$R" -eq 0 ]; then echo "SYNTHETIC: PASS"; else echo "SYNTHETIC: FAIL (rc=$R)"; RC=1; fi

# ---- 3. real genes: parity + timing --------------------------------------
TSV="$OUT/nj_gpu0_${SLURM_JOB_ID:-local}.tsv"
echo; echo "### real genes -> $TSV ###"
first=1
for FA in "$GENES_DIR"/*.fasta; do
    G=$(basename "$FA" .fasta)
    if [ $first -eq 1 ]; then HDR=--header; first=0; else HDR=; fi
    "$BIN/nj_bench" "$FA" --name "$G" --threads "$T" $HDR >> "$TSV" || RC=1
done
echo; column -t "$TSV" 2>/dev/null || cat "$TSV"
echo
NFAIL=$(awk 'NR>1 && $10!="PASS"' "$TSV" | wc -l)
NGENE=$(awk 'NR>1' "$TSV" | wc -l)
if [ "$NFAIL" -eq 0 ] && [ "$NGENE" -gt 0 ]; then
    echo "REAL-GENE PARITY: PASS ($NGENE genes, Tree.nodes identical)"
else
    echo "REAL-GENE PARITY: FAIL ($NFAIL of $NGENE genes)"; RC=1
fi

# ---- 3b. per-kernel profile on one gene (rocprof, optional) -----------------
ROCPROF=$(dirname "$HIPCC")/rocprof
if [ -x "$ROCPROF" ] && [ -s "$GENES_DIR/$GENE.fasta" ]; then
    echo; echo "### rocprof --stats on $GENE ###"
    ( cd "$BIN" && "$ROCPROF" --stats -o "$BIN/prof.csv" \
        "$BIN/nj_bench" "$GENES_DIR/$GENE.fasta" --name "$GENE" --no-cpu >/dev/null 2>&1 )
    [ -s "$BIN/prof.stats.csv" ] && { column -s, -t "$BIN/prof.stats.csv" | head -8; \
        cp "$BIN/prof.stats.csv" "$OUT/nj_gpu0_${SLURM_JOB_ID:-local}_${GENE}.stats.csv"; }
fi

# ---- 4. end-to-end: genomsa with GPU tree vs CPU tree --------------------
FG="$GENES_DIR/$GENE.fasta"
if [ -s "$FG" ]; then
    echo; echo "### genomsa end-to-end on $GENE: GPU tree vs CPU tree ###"
    "$BIN/genomsa" "$FG" "$BIN/${GENE}.gputree.aln" 2>&1 | tail -3
    GENOMSA_NJ=cpu "$BIN/genomsa" "$FG" "$BIN/${GENE}.cputree.aln" 2>&1 | tail -3
    if cmp -s "$BIN/${GENE}.gputree.aln" "$BIN/${GENE}.cputree.aln"; then
        echo "E2E: PASS (alignment byte-identical with GPU and CPU guide tree)"
    else
        echo "E2E: FAIL (alignment differs)"; RC=1
    fi
fi

echo; echo "=== done $(date) rc=$RC ==="
exit $RC
