#!/bin/bash
#SBATCH --job-name=codon_smoke
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/data/codon_smoke/slurm-%A_%a.log
#
# Smoke benchmark for genomsa --codon vs MACSE(-init) vs PRANK-codon vs
# 3-step (MAFFT-AA back-translate) on simulated truth.
#
# Array layout: task = cell_idx * REPS + rep  (cells x reps from
# scripts/codon_smoke_grid.json).
#
# Submit:  sbatch --array=0-$((NCELLS*REPS-1)) scripts/codon_smoke.sbatch

set -u
PAI=/beegfs/a474r867/phylogenyAI
REPO=/beegfs/a474r867/genoaligner/repo
GRID=$REPO/scripts/codon_smoke_grid.json
OUT=$PAI/data/codon_smoke
BIN=$OUT/bin
ENV=/home/a474r867/miniconda3/envs/genoml-phylo
mkdir -p "$OUT" "$BIN"
ROCM=/kuhpc/sw/rocm/6.4.3

# --- build genomsa once per node (cached; rebuild if binary missing) ------
if [ ! -x "$BIN/genomsa" ]; then
    "$ROCM/bin/hipcc" -O2 -std=c++17 -ffp-contract=off -I"$REPO/include" \
        -o "$BIN/genomsa" "$REPO/tools/genomsa.cpp" \
        "$REPO/src/msa/msa_gpu.cpp" "$REPO/src/msa/msa_ref.cpp" \
        "$REPO/src/msa/nj_gpu.cpp" 2>>"$OUT/build.log"
fi

export PATH="$ENV/bin:$PATH"
MACSE="$ENV/bin/macse"
PRANK="$ENV/bin/prank"

# --- resolve (cell, rep) ---------------------------------------------------
REPS=$(python3 -c "import json;print(json.load(open('$GRID'))['reps'])")
TASK=${SLURM_ARRAY_TASK_ID:-0}
CELL_IDX=$((TASK / REPS))
REP=$((TASK % REPS))

SPEC=$(python3 - <<EOF
import json
g = json.load(open("$GRID"))
d = dict(g["defaults"]); d.update(g["cells"][$CELL_IDX])
args = []
for k in ("n","L","gc","sub","indel","omega","fs_frac","stop_frac",
          "frag5","frag3","err","lr_frac"):
    if k in d:
        flag = k.replace("_","-")
        args.append(f"--{flag} {d[k]}")
print(d["id"] + "|" + " ".join(args))
EOF
)
CELL=${SPEC%%|*}
CARGS=${SPEC#*|}

WD=$OUT/$CELL/r$REP
mkdir -p "$WD"

# Per-instance file: parallel array tasks must not interleave writes.
python3 "$REPO/scripts/codon_bench_grid.py" "$WD" \
    --cell "$CELL" --rep "$REP" $CARGS \
    --tools genomsa,macse,prank,threestep \
    --genomsa "$BIN/genomsa" --macse "$MACSE" --prank "$PRANK" \
    --timeout 5400 \
    > "$WD/results.tsv" 2>"$WD/stderr.log"
