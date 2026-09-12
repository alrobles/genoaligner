#!/bin/bash -l
#SBATCH --job-name=h8_b6_pro6000
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:pro6000:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h8b6pro6000_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h8b6pro6000_%j.err

# Production flag B6 — throughput numbers with error bars.
#
# WHY THIS IS A SEPARATE DESIGN AND NOT "RUN THE BENCH AGAIN"
# ----------------------------------------------------------
# Fase 6 reported single numbers produced by separate jobs. Fase 7 then measured
# node-to-node variance of 3-4x on identical input and a kernel that varied 4.4x
# between launches WITHIN one process, which makes any cross-job number arithmetic on
# a moving baseline. An early "1.93x" was published and retracted for exactly that
# reason.
#
# So B6 measures inside ONE job per GPU, and for every regime:
#   - R independent repetitions of the SAME configuration (spread within the job),
#   - both kernels interleaved A,B,A,B so any drift inside the job hits both, and the
#     RATIO is what gets reported as the comparison,
#   - the statistics (mean, sd, min, max) printed, not just a mean.
#
# The metric is kernel-only TCUPS, the one comparable to Accelign/WFA-GPU, computed
# by the benchmark over the cells of RESOLVED pairs.
#
# Usage: sbatch scripts/h8_b6_pro6000_tcus.sbatch            (MI210, four regimes, 7 reps)
set -o pipefail

ROOT=/beegfs/a474r867/genoaligner/repo
OUT_DIR=/beegfs/a474r867/genoaligner/bench
LOGD=/beegfs/a474r867/genoaligner/logs
mkdir -p "$OUT_DIR" "$LOGD"

source /kuhpc/sw/lmod/lmod/init/profile 2>/dev/null
ml purge >/dev/null 2>&1
cd "$ROOT" || exit 1

REPS=${REPS:-7}
BIN=/tmp/b6_bench_pro6000

echo "=== host: $(hostname) ==="
echo "=== H8-B6: kernel-only TCUPS with error bars (RTX PRO 6000) ==="
echo "  reps per regime: $REPS   (independent runs, same configuration)"
echo "  interleaved: default/flat alternate so in-job drift hits both, ratio reported"
echo

C=/kuhpc/sw/nvhpc/Linux_x86_64/25.3/cuda/12.8
"$C/bin/nvcc" -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
    -gencode arch=compute_120,code=sm_120 \
    -DWFA_FLAT_BLOCK=128 -DGENOALIGNER_BENCH_BACKEND='"cuda"' \
    -I /kuhpc/sw/rocm/6.4.3/include -I "$C/include" \
    -I "$C/targets/x86_64-linux/include" -I . -I include \
    -lcuda -L "$C/lib64/stubs" -x cu -o "$BIN" bench/tcus.cpp
RC=$?
echo "BUILD_RC=$RC"
[ "$RC" -ne 0 ] && { echo "H8-B6: BUILD FAILED"; exit 1; }

echo
echo "device: $("$BIN" --len 256 --pairs 64 --ident 99 --smax 64 --reps 1 --verify 5 2>&1 | grep -m1 'device')"

# Extract the kernel-only TCUPS from one run. Returns empty if the run did not verify
# (exit 5/6) -- a throughput from an unverified run is not a throughput.
kernel_tcus() {
    local out rc
    out=$("$BIN" "$@" --reps 20 --verify 50 2>&1)
    rc=$?
    if [ "$rc" -ne 0 ]; then echo ""; return; fi
    echo "$out" | grep -E 'kernel-only' | sed 's/.*: *//; s/ *TCUPS.*//'
}

# Stats over a whitespace-separated list on stdin: n mean sd min max
stats() {
    awk '{for(i=1;i<=NF;i++) v[n++]=$i}
         END{
           if(n==0){print "n/a"; exit}
           s=0; for(i=0;i<n;i++) s+=v[i]; m=s/n
           q=0; for(i=0;i<n;i++) q+=(v[i]-m)*(v[i]-m)
           sd=(n>1)?sqrt(q/(n-1)):0
           mn=v[0]; mx=v[0]
           for(i=1;i<n;i++){ if(v[i]<mn)mn=v[i]; if(v[i]>mx)mx=v[i] }
           printf "n=%d  mean=%.4f  sd=%.4f  min=%.4f  max=%.4f  spread=%.1f%%\n",
                  n, m, sd, mn, mx, (mn>0)?100.0*(mx-mn)/mn:0
         }'
}

RESULT="$OUT_DIR/h8b6_${SLURM_JOB_ID}.txt"
: > "$RESULT"

# Four regimes: the Fase 6 set, all of which resolve fully at the given identity.
REGIMES=(
  "256 8000 90 64|len=256  smax=64  ident=90%"
  "1024 2000 90 256|len=1024 smax=256 ident=90%"
  "1024 2000 70 256|len=1024 smax=256 ident=70%"
  "2048 300 95 128|len=2048 smax=128 ident=95%"
)

for R in "${REGIMES[@]}"; do
    SPEC="${R%%|*}"; LABEL="${R##*|}"
    set -- $SPEC
    LEN=$1; PAIRS=$2; IDENT=$3; SMAX=$4
    echo
    echo "######## $LABEL"
    D_LIST=""; F_LIST=""

    printf "%6s %14s %14s %10s\n" "rep" "default TCUPS" "flat TCUPS" "ratio"
    for ((i=1;i<=REPS;i++)); do
        D=$(kernel_tcus --len $LEN --pairs $PAIRS --ident $IDENT --smax $SMAX --kernel default)
        F=$(kernel_tcus --len $LEN --pairs $PAIRS --ident $IDENT --smax $SMAX --kernel flat)
        if [ -z "$D" ] || [ -z "$F" ]; then
            printf "%6d %14s %14s %10s\n" "$i" "VERIFY-FAIL" "VERIFY-FAIL" "-"
            continue
        fi
        RAT=$(awk -v d="$D" -v f="$F" 'BEGIN{ if(f>0) printf "%.3f", d/f; else print "n/a" }')
        printf "%6d %14s %14s %10s\n" "$i" "$D" "$F" "$RAT"
        D_LIST="$D_LIST $D"; F_LIST="$F_LIST $F"
    done

    echo "  default : $(echo $D_LIST | stats)"
    echo "  flat    : $(echo $F_LIST | stats)"
    RATIOS=$(awk -v d="$D_LIST" -v f="$F_LIST" 'BEGIN{
        split(d,D," "); split(f,F," "); n=0
        for(i=1;i<=length(D)&&i<=length(F);i++) if(D[i]!=""&&F[i]>0){printf "%.4f ", D[i]/F[i]; n++}
    }')
    echo "  ratio   : $(echo $RATIOS | stats)"
    {
        echo "### $LABEL"
        echo "default:$D_LIST"
        echo "flat:$F_LIST"
        echo "ratio:$RATIOS"
    } >> "$RESULT"
done

echo
echo "=== H8-B6 DONE ==="
echo "numbers also written to $RESULT"