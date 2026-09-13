#!/bin/bash -l
#SBATCH --job-name=msa_baselines
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/msabase_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/msabase_%j.err
#
# Established CPU MSA baselines on the same bench1.0 inputs genomsa uses.
# Per-family wall-clock, one thread per invocation, families parallelized
# across the node. Output: $BASE/<method>_out/<set>/<family> + timings TSV.
set -o pipefail
E=/home/a474r867/miniconda3/envs/genoml-phylo/bin
BENCH=/beegfs/a474r867/bench/bench1.0
BASE=/beegfs/a474r867/bench
REPO=/beegfs/a474r867/genoaligner/repo
TIMES=$BASE/baseline_times.tsv
echo "=== host $(hostname) $(date) ==="
$E/mafft --version 2>&1 | head -1
$E/muscle -version 2>&1 | head -1
$E/kalign --version 2>&1 | head -1

run_one() {   # method dataset family
    m=$1; d=$2; f=$3
    in="$BENCH/$d/in/$f"; out="$BASE/${m}_out/$d/$f"
    [ -s "$out" ] && return 0
    s=$EPOCHREALTIME
    case $m in
        mafft)  $E/mafft --auto --thread 1 --quiet "$in" > "$out" 2>/dev/null ;;
        linsi)  $E/mafft --localpair --maxiterate 1000 --thread 1 --quiet \
                    "$in" > "$out" 2>/dev/null ;;
        muscle) $E/muscle -align "$in" -output "$out" -threads 1 \
                    >/dev/null 2>&1 ;;
        kalign) $E/kalign -i "$in" -o "$out" -n 1 >/dev/null 2>&1 ;;
    esac
    rc=$?
    e=$EPOCHREALTIME
    ok=ok; { [ $rc -ne 0 ] || [ ! -s "$out" ]; } && ok=FAIL
    printf "%s\t%s\t%s\t%.3f\t%s\n" "$m" "$d" "$f" \
        "$(echo "$e - $s" | bc)" "$ok" >> "$TIMES"
}
export -f run_one
export E BENCH BASE TIMES

# method list per dataset: linsi only on bali3+sabrem (O(n) iterative)
for d in bali3 sabrem ox prefab4; do
    for m in mafft muscle kalign; do
        mkdir -p "$BASE/${m}_out/$d"
        echo "=== $m $d ==="
        ls "$BENCH/$d/in" | xargs -P 32 -I{} bash -c 'run_one "$0" "$1" "$2"' \
            "$m" "$d" {}
        grep -c "	${d}	" "$TIMES" >/dev/null
        echo "done $m $d"
    done
done
for d in bali3 sabrem; do
    m=linsi; mkdir -p "$BASE/${m}_out/$d"
    echo "=== linsi $d ==="
    ls "$BENCH/$d/in" | xargs -P 32 -I{} bash -c 'run_one "$0" "$1" "$2"' \
        "$m" "$d" {}
    echo "done linsi $d"
done
echo "BASELINES DONE"
