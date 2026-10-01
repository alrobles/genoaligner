#!/bin/bash -l
#SBATCH --job-name=codon_sim
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/codonsim_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/codonsim_%j.err

# Codon-mode quality benchmark on simulated truth:
#   msa_sim_truth.py (codon mode: sense-codon ancestor, whole-codon indels)
#   -> genomsa --codon AND macse on the same input
#   -> column-SPS vs the true MSA for each
# This is the honest quality gate: agreement-with-MACSE on real data is
# ambiguous, SIM-SPS against known truth is not.
set -o pipefail
REPO=${REPO:-/beegfs/a474r867/genoaligner/repo}
ROCM=/kuhpc/sw/rocm/6.4.3
OUT=/beegfs/a474r867/phylogenyAI/data/codon_sim
mkdir -p "$OUT"; cd "$REPO" || exit 1
source /beegfs/a474r867/miniconda3/etc/profile.d/conda.sh
conda activate genoml-phylo

echo "=== $(hostname) $(date) job ${SLURM_JOB_ID:-local} ==="; git log --oneline -1
BIN=/tmp/codonsim_${SLURM_JOB_ID:-local}; mkdir -p "$BIN"
"$ROCM/bin/hipcc" -O2 -ffp-contract=off -std=c++17 -I. -Iinclude \
    -o "$BIN/genomsa" tools/genomsa.cpp src/msa/msa_gpu.cpp \
    src/msa/nj_gpu.cpp src/msa/msa_ref.cpp || exit 1

printf "cell\tn\tL\tsub\tindel\tgenomsa_sps\tgenomsa_s\tmacse_sps\tmacse_s\n" > "$OUT/grid_${SLURM_JOB_ID:-local}.tsv"
cell=0
for SUB in 0.02 0.06 0.10; do
  for IND in 0.005 0.02; do
    cell=$((cell+1)); W=$BIN/c$cell; mkdir -p "$W"; cd "$W"
    python3 "$REPO/scripts/msa_sim_truth.py" - 256 1 "$SUB" "$IND" 1200 codon >/dev/null
    t0=$SECONDS
    "$BIN/genomsa" sim_in.fasta sim_got.fasta --codon 2>g.log || { echo "genomsa FAIL c$cell"; continue; }
    tg=$((SECONDS - t0))
    GS=$(python3 "$REPO/scripts/msa_sim_truth.py" sim_got.fasta 256 1 "$SUB" "$IND" 1200 codon | grep -o 'SIM-SPS [0-9.]*' | awk '{print $2}')
    t0=$SECONDS
    macse -prog alignSequences -seq sim_in.fasta -out_NT sim_macse.fasta \
        -out_AA sim_macse_aa.fasta > m.log 2>&1 \
        || { echo "macse FAIL c$cell"; MS=NA; tm=-1; }
    tm=$((SECONDS - t0))
    if [ "${MS:-x}" != NA ]; then
        MS=$(python3 "$REPO/scripts/msa_sim_truth.py" sim_macse.fasta 256 1 "$SUB" "$IND" 1200 codon | grep -o 'SIM-SPS [0-9.]*' | awk '{print $2}')
    fi
    printf "c%s\t256\t1200\t%s\t%s\t%s\t%s\t%s\t%s\n" "$cell" "$SUB" "$IND" "${GS:-NA}" "$tg" "${MS:-NA}" "$tm" | tee -a "$OUT/grid_${SLURM_JOB_ID:-local}.tsv"
    cd "$REPO"
  done
done
echo; column -t "$OUT/grid_${SLURM_JOB_ID:-local}.tsv"
