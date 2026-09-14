#!/bin/bash
#SBATCH --job-name=codon_smoke_cpu
#SBATCH --partition=sixhour
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/beegfs/a474r867/phylogenyAI/data/codon_smoke/slurm-cpu-%A_%a.log
#
# CPU-only variant of codon_smoke: genomsa --cpu is byte-identical to the
# GPU path (proven on MI210 + A100), so the full tool matrix can run on
# any sixhour node without GPU contention. GPU walltime for genomsa is
# measured separately on a few instances (codon_smoke.sbatch).
#
# Array: task = cell_idx * REPS + rep.

set -u
PAI=/beegfs/a474r867/phylogenyAI
REPO=/beegfs/a474r867/genoaligner/repo
GRID=$REPO/scripts/codon_smoke_grid.json
OUT=${CODON_SMOKE_OUT:-$PAI/data/codon_smoke}
BIN=$OUT/bin
ENV=/home/a474r867/miniconda3/envs/genoml-phylo
mkdir -p "$OUT" "$BIN"

# CPU-shim build: no HIP runtime needed at run time. Lock to avoid the
# array stampede compiling at once; build to a temp path and mv so the
# -x test never sees a half-linked binary (caused EACCES on v2.1 run).
if [ ! -x "$BIN/genomsa_cpu" ]; then
    (
        flock -x 9
        if [ ! -x "$BIN/genomsa_cpu" ]; then
            g++ -O2 -std=c++17 -DGENOALIGNER_HIP_SHIM \
                -I"$REPO/tests/parity/hip_cpu_shim" -I"$REPO/include" \
                -pthread "$REPO/tools/genomsa.cpp" \
                "$REPO/src/msa/msa_gpu.cpp" "$REPO/src/msa/msa_ref.cpp" \
                "$REPO/src/msa/nj_gpu.cpp" \
                -o "$BIN/genomsa_cpu.tmp.$$" 2>>"$OUT/build_cpu.log" \
                && mv "$BIN/genomsa_cpu.tmp.$$" "$BIN/genomsa_cpu"
        fi
    ) 9>"$OUT/build.lock"
fi

export PATH="$ENV/bin:$PATH"
MACSE="$ENV/bin/macse"
PRANK="$ENV/bin/prank"

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

python3 "$REPO/scripts/codon_bench_grid.py" "$WD" \
    --cell "$CELL" --rep "$REP" $CARGS \
    --tools genomsa,macse,prank,threestep --prank-F \
    --genomsa "$BIN/genomsa_cpu" --genomsa-cpu \
    --macse "$MACSE" --prank "$PRANK" \
    --timeout 5400 \
    > "$WD/results.tsv" 2>"$WD/stderr.log"
