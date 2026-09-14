#!/bin/bash -l
#SBATCH --job-name=nj_trace
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/njtrace_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/njtrace_%j.err

# NJ candidate-recall trace (docs/SPEC_NJ_GPU.md, NJ1 gate). CPU only.
#
#   1. build tools/nj_trace with g++ (no HIP)
#   2. for every gene in $PAI/data/genes_qc_pass: exact NJ + per-round
#      Recall@k of the cherry in fresh/static top-k candidates + fraction
#      of rounds certifiable by the row-wise bound
#   3. one TSV line per gene -> $OUT/nj_trace_summary.tsv, per-round
#      traces -> $OUT/rounds/<gene>.tsv
#
# REPO is overridable:  sbatch --export=REPO=/path/to/checkout ...
set -o pipefail

PAI=/beegfs/a474r867/phylogenyAI
REPO=${REPO:-/beegfs/a474r867/genoaligner/repo}
OUT=${OUT_DIR:-$PAI/data/nj_trace}
THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "$OUT/rounds" "$OUT/trees"
cd "$REPO" || exit 1

echo "=== host: $(hostname)  $(date)  commit: $(git rev-parse --short HEAD 2>/dev/null) ==="
g++ -O2 -std=c++17 -Iinclude -o /tmp/nj_trace tools/nj_trace.cpp src/msa/msa_ref.cpp -lpthread \
    || { echo "BUILD FAILED"; exit 1; }
echo "build ok"

SUMMARY=$OUT/nj_trace_summary.tsv
: > "$SUMMARY"
first=1
for fa in "$PAI"/data/genes_qc_pass/*.fasta "$PAI"/data/genes_qc_pass/*.fa; do
    [ -f "$fa" ] || continue
    gene=$(basename "$fa"); gene=${gene%.*}
    n=$(grep -c '^>' "$fa")
    echo "--- $gene n=$n $(date +%T)"
    args=(--name "$gene" --threads "$THREADS" --rounds "$OUT/rounds/$gene.tsv" --tree "$OUT/trees/$gene.tsv")
    [ $first -eq 1 ] && args+=(--header)
    if /tmp/nj_trace "$fa" "${args[@]}" >> "$SUMMARY"; then
        first=0
    else
        echo "FAILED $gene rc=$?"
    fi
done

echo "=== summary ==="
column -t "$SUMMARY"
echo "=== done $(date) ==="
