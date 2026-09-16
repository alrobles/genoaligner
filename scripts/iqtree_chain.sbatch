#!/bin/bash -l
#SBATCH --job-name=iqt_chain
#SBATCH --partition=sixhour
#SBATCH --exclude=r06r26n01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/logs/iqt_chain_%j.out
#SBATCH --error=/beegfs/a474r867/phylogenyAI/logs/iqt_chain_%j.err
# Self-chaining IQ-TREE: runs <=6h links, resumes from .ckp.gz each time.
# Export vars: SM (supermatrix dir), SCHEME (.nex), SEED, PRE (prefix),
#              SEEDTREE (optional -t start tree), BOOT (default 1000),
#              MAXLINKS (default 40)
set -uo pipefail
source /beegfs/a474r867/miniconda3/etc/profile.d/conda.sh
conda activate genoml-phylo

OUT=$(dirname "$PRE"); mkdir -p "$OUT"
CNT_FILE=$OUT/chain_count_${SEED:-1}
N=$(cat "$CNT_FILE" 2>/dev/null || echo 0)
echo "chain link $N (seed=${SEED:-1}) jobid=$SLURM_JOB_ID"

# SIGINT 20min before walltime -> iqtree writes a clean .ckp.gz
timeout -s INT -k 120 20400 iqtree2 \
    -s "$SM/supermatrix.fasta" -p "$SCHEME" \
    ${SEEDTREE:+-t "$SEEDTREE"} \
    -B "${BOOT:-1000}" -seed "${SEED:-1}" \
    -T 16 -ntmax 16 -pre "$PRE" --quiet || true

if [ -s "$PRE.contree" ]; then
    echo "CHAIN DONE at link $N"
    grep -m1 'Log-likelihood of the tree' "$PRE.iqtree" 2>/dev/null
    exit 0
fi
N=$((N+1))
if [ "$N" -ge "${MAXLINKS:-40}" ]; then
    echo "CHAIN LIMIT ($N links) reached without .contree"; exit 1
fi
# fast-fail guard: a link that left no checkpoint/log never ran iqtree
if [ ! -s "$PRE.ckp.gz" ] && [ ! -s "$PRE.log" ]; then
    echo "iqtree produced no output (instant fail) -- aborting chain"; exit 1
fi
echo "$N" > "$CNT_FILE"
sbatch --export=ALL,SM="$SM",SCHEME="$SCHEME",SEED="${SEED:-1}",PRE="$PRE",\
SEEDTREE="${SEEDTREE:-}",BOOT="${BOOT:-1000}",MAXLINKS="${MAXLINKS:-40}" "$0"
echo "resubmitted link $N"
