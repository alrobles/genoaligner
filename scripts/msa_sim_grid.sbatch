#!/bin/bash -l
#SBATCH --job-name=genomsa_simgrid
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/msasim_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/msasim_%j.err
#
# Synthetic scaling grid for genomsa: simulated families with known true
# alignments (scripts/msa_sim_truth.py). Varies nseq x seqlen, divergence
# and indel rate. genomsa --cpu timed per cell; MAFFT --auto as baseline.
set -o pipefail
REPO=/beegfs/a474r867/genoaligner/repo
E=/home/a474r867/miniconda3/envs/genoml-phylo/bin
BASE=/beegfs/a474r867/bench/simgrid
mkdir -p "$BASE"
cd "$REPO" || exit 1
echo "=== host $(hostname) $(date) ==="

g++ -O2 -std=c++17 -pthread -DGENOALIGNER_HIP_SHIM \
    -I tests/parity/hip_cpu_shim -I include \
    -o /tmp/genomsa_sim tools/genomsa.cpp src/msa/msa_gpu.cpp \
    src/msa/msa_ref.cpp || { echo BUILD_FAILED; exit 1; }
echo "build ok"

echo -e "cell\tn\tL\tsub\tindel\tgenomsa_s\tgenomsa_sps\tmafft_s\tmafft_sps" \
    > "$BASE/grid.tsv"

cell() {   # n L sub indel
    n=$1; L=$2; sub=$3; ind=$4; tag="n${n}_L${L}_s${sub}_i${ind}"
    w="$BASE/$tag"; mkdir -p "$w"; cd "$w" || return
    python3 "$REPO/scripts/msa_sim_truth.py" - "$n" 1 "$sub" "$ind" "$L" \
        >/dev/null || { echo "$tag SIMFAIL"; cd "$REPO"; return; }
    s=$EPOCHREALTIME
    /tmp/genomsa_sim sim_in.fasta got_genomsa.fa --cpu >/dev/null 2>&1
    rc1=$?
    e=$EPOCHREALTIME; tg=$(echo "$e - $s" | bc)
    spsg="-"; [ $rc1 -eq 0 ] && spsg=$(python3 \
        "$REPO/scripts/msa_sim_truth.py" got_genomsa.fa "$n" 1 "$sub" "$ind" \
        "$L" 2>/dev/null | grep -o 'SIM-SPS [0-9.]*' | awk '{print $2}')
    s=$EPOCHREALTIME
    $E/mafft --auto --thread 8 --quiet sim_in.fasta > got_mafft.fa 2>/dev/null
    rc2=$?
    e=$EPOCHREALTIME; tm=$(echo "$e - $s" | bc)
    spsm="-"; [ $rc2 -eq 0 ] && spsm=$(python3 \
        "$REPO/scripts/msa_sim_truth.py" got_mafft.fa "$n" 1 "$sub" "$ind" \
        "$L" 2>/dev/null | grep -o 'SIM-SPS [0-9.]*' | awk '{print $2}')
    printf "%s\t%d\t%d\t%s\t%s\t%.2f\t%s\t%.2f\t%s\n" \
        "$tag" "$n" "$L" "$sub" "$ind" "$tg" "$spsg" "$tm" "$spsm" \
        | tee -a "$BASE/grid.tsv"
    cd "$REPO" || return
}

# scaling cells (bounded product n*L); divergence/indel sweep at n=100,L=1000
cell 10    100   0.02 0.004
cell 10    1000  0.02 0.004
cell 10    10000 0.02 0.004
cell 100   100   0.02 0.004
cell 100   1000  0.02 0.004
cell 100   10000 0.02 0.004
cell 1000  100   0.02 0.004
cell 1000  1000  0.02 0.004
cell 100   1000  0.10 0.004
cell 100   1000  0.30 0.004
cell 100   1000  0.02 0.020
cell 100   1000  0.02 0.080
cell 10000 100   0.02 0.004
echo "SIMGRID DONE"
