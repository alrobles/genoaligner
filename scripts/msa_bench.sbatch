#!/bin/bash -l
#SBATCH --job-name=genomsa_bench
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/msabench_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/msabench_%j.err
#
# genomsa protein benchmark: BAliBASE v3 / PREFAB v4 / SABRE(SABmark1.65) /
# OXBench from the drive5 bench1.0 bundle. Scoring: SP(Q)/TC on core
# (uppercase) columns, qscore-compatible (scripts/msa_bench_score.py).
set -o pipefail
REPO=/beegfs/a474r867/genoaligner/repo
BENCH=/beegfs/a474r867/bench/bench1.0
OUT=/beegfs/a474r867/bench/genomsa_out
mkdir -p "$OUT"
cd "$REPO" || exit 1
echo "=== host: $(hostname) $(date) ==="

# CPU-shim build: --cpu path needs no device; same source as GPU build.
g++ -O2 -std=c++17 -DGENOALIGNER_HIP_SHIM \
    -I tests/parity/hip_cpu_shim -I include \
    -o /tmp/genomsa_bench tools/genomsa.cpp src/msa/msa_gpu.cpp \
    src/msa/msa_ref.cpp src/io/fasta.cpp 2>/dev/null || \
g++ -O2 -std=c++17 -DGENOALIGNER_HIP_SHIM \
    -I tests/parity/hip_cpu_shim -I include \
    -o /tmp/genomsa_bench tools/genomsa.cpp src/msa/msa_gpu.cpp \
    src/msa/msa_ref.cpp || { echo BUILD_FAILED; exit 1; }
echo "build ok"

TIMES=/beegfs/a474r867/bench/genomsa_cpu_times.tsv
align_one() {
    d=$1; f=$2
    s=$EPOCHREALTIME
    /tmp/genomsa_bench "$BENCH/$d/in/$f" "$OUT/$d/$f" --cpu --protein \
        >/dev/null 2>&1
    rc=$?
    e=$EPOCHREALTIME
    ok=ok; { [ $rc -ne 0 ] || [ ! -s "$OUT/$d/$f" ]; } && ok=FAIL
    printf "genomsa_cpu\t%s\t%s\t%.3f\t%s\n" "$d" "$f" \
        "$(echo "$e - $s" | bc)" "$ok" >> "$TIMES"
}
export -f align_one
export BENCH OUT TIMES

for d in bali3 sabrem ox prefab4; do
    mkdir -p "$OUT/$d"
    echo "=== $d: aligning $(ls $BENCH/$d/in | wc -l) families ==="
    ls "$BENCH/$d/in" | xargs -P 32 -I{} bash -c 'align_one "$0" "$1"' "$d" {}
    echo "=== $d: scoring ==="
    python3 "$REPO/scripts/msa_bench_score.py" --scan "$OUT/$d" "$BENCH/$d/ref" \
        > "$OUT/$d.scores.tsv" 2>"$OUT/$d.err"
    tail -1 "$OUT/$d.scores.tsv"
done
echo "BENCH DONE"
