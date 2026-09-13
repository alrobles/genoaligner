#!/bin/bash -l
#SBATCH --job-name=tw_vs_gsa
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/twbench_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/twbench_%j.err

# Head-to-head: TWILIGHT (CUDA, V100) vs genomsa (wavefront, V100) on the
# same V100 node. Two axes:
#   (a) quality  -- column-SPS vs the known simulated truth (n=64)
#   (b) speed    -- wall time on real genes (COI, CYTB)
set -o pipefail
PAI=/beegfs/a474r867/phylogenyAI
REPO=/beegfs/a474r867/genoaligner/repo
B=/beegfs/a474r867/twilight_bench
TWB=/beegfs/a474r867/twilight/twilight-v100
R=/kuhpc/sw/rocm/6.4.3
C=/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4

T() { local s=$SECONDS; "$@"; local rc=$?; echo "WALL $((SECONDS-s))s rc=$rc"; return $rc; }

echo "=== host: $(hostname) $(date) ==="
nvidia-smi -L | head -2
[ -x "$TWB" ] || { echo "missing twilight binary $TWB"; exit 1; }

"$C/bin/nvcc" -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
    -gencode arch=compute_70,code=sm_70 \
    -I "$R/include" -I "$C/include" -I "$C/targets/x86_64-linux/include" \
    -I "$REPO" -I "$REPO/include" -lcuda -L "$C/lib64/stubs" -x cu \
    -o /tmp/genomsa "$REPO/tools/genomsa.cpp" "$REPO/src/msa/msa_gpu.cpp" "$REPO/src/msa/msa_ref.cpp" \
    || { echo "genomsa build failed"; exit 1; }
echo "genomsa build ok"

cd "$B" || exit 1
echo; echo "########## SIM64 quality (same input, two guide trees) ##########"

echo "--- TWILIGHT, genomsa NJ guide tree ---"
T "$TWB" -t sim64_njtree.nwk -i sim64_in.fasta -o tw_nj.aln --type n --overwrite 2>&1 | tail -6

echo "--- TWILIGHT, TRUE guide tree ---"
T "$TWB" -t sim64_truetree.nwk -i sim64_in.fasta -o tw_true.aln --type n --overwrite 2>&1 | tail -6

echo "--- TWILIGHT, NJ tree, gappy removal OFF (-r 1) ---"
T "$TWB" -t sim64_njtree.nwk -i sim64_in.fasta -o tw_nj_r1.aln --type n -r 1 --overwrite 2>&1 | tail -6

echo "--- genomsa (own NJ tree, wavefront) ---"
T /tmp/genomsa sim64_in.fasta gsa_sim64.aln 2>&1 | tail -6

echo; echo "--- SPS vs truth ---"
for a in tw_nj tw_true tw_nj_r1 gsa_sim64; do
    if [ -s "$a.aln" ]; then
        echo -n "$a: "
        python3 "$REPO/scripts/msa_sim_truth.py" "$a.aln" 64 4 0.02 0.004
    else echo "$a: MISSING"; fi
done

echo; echo "########## REAL GENES (speed, V100 both) ##########"
for GENE in COI CYTB; do
    FA="$PAI/data/genes_qc_pass/${GENE}.fasta"
    [ -s "$FA" ] || continue
    echo "=== $GENE ==="
    T /tmp/genomsa "$FA" "$B/${GENE}.gsa.aln" --tree-out "$B/${GENE}.nwk" 2>&1 | tail -5
    T "$TWB" -t "$B/${GENE}.nwk" -i "$FA" -o "$B/${GENE}.tw.aln" --type n --overwrite 2>&1 | tail -5
done

echo; echo "TW-BENCH: DONE"
