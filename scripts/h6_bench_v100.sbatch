1|#!/bin/bash -l
2|#SBATCH --job-name=h6_bench_v100
3|#SBATCH --partition=sixhour
4|#SBATCH --nodes=1
5|#SBATCH --ntasks-per-node=1
6|#SBATCH --cpus-per-task=8
7|#SBATCH --mem=64G
8|#SBATCH --time=00:30:00
9|#SBATCH --gres=gpu:v100:1
10|#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h6v100_%j.out
11|#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h6v100_%j.err
12|
# Fase 6 on the V100 (sm_70, Volta). Same bench/tcus.cpp, same regimes, CUDA 12.4.
# Volta is the oldest architecture this site's nvcc can target and has the weakest
# memory subsystem of the set, so it is the floor of the comparison.
#
# Twins: h6_bench_{cuda,a100,l40}.sbatch -- identical except --gres and -gencode.
19|set -o pipefail
20|
21|R=/kuhpc/sw/rocm/6.4.3
22|C=/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4
23|ARCH=${ARCH:-compute_70,code=sm_70}
24|ROOT=/beegfs/a474r867/genoaligner/repo
25|OUT_DIR=/beegfs/a474r867/genoaligner/bench
26|mkdir -p "$OUT_DIR"
27|BIN=/tmp/tcus_v100
28|
29|echo "=== host: $(hostname) ==="
30|echo "=== H6-V100: Fase 6 timing probe (V100, 70) ==="
31|echo
32|
33|nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
34|"$C/bin/nvcc" --version | tail -2
35|
36|cd "$ROOT" || exit 1
37|
38|echo
39|echo "--- build (nvcc 12.4, $ARCH) ---"
40|"$C/bin/nvcc" -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
41|    -gencode arch=$ARCH \
42|    -DGENOALIGNER_BENCH_BACKEND='"cuda"' \
43|    -I "$R/include" \
44|    -I "$C/include" \
45|    -I "$C/targets/x86_64-linux/include" \
46|    -I . \
47|    -lcuda -L "$C/lib64/stubs" \
48|    -x cu -o "$BIN" bench/tcus.cpp
49|RC_BUILD=$?
50|echo "BUILD_RC=$RC_BUILD"
51|if [ "$RC_BUILD" -ne 0 ]; then echo "H6-V100: BUILD FAILED"; exit 1; fi
52|
53|"$C/bin/cuobjdump" --list-elf "$BIN" 2>/dev/null | head -3
54|
55|rc_all=0
56|rc_verified=0
57|run() {
58|    local label="$1"; shift
59|    echo
60|    echo "=============================================================="
61|    echo "REGIME: $label"
62|    echo "=============================================================="
63|    "$BIN" "$@" | tee -a "$OUT_DIR/h6_v100_${SLURM_JOB_ID}.txt"
64|    local rc=${PIPESTATUS[0]}
65|    case "$rc" in
66|        0) : ;;
67|        5) echo ">>> regime UNVERIFIED (rc=5): timing measured nothing."; rc_all=1 ;;
68|        6) echo ">>> regime INVALID (rc=6): kernel wrong."; rc_all=1 ;;
69|        *) echo ">>> regime FAILED (rc=$rc)."; rc_all=1 ;;
70|    esac
71|    [ "$rc" -eq 0 ] && rc_verified=$((rc_verified + 1))
72|    return 0
73|}
74|
75|# Identical regimes to the MI210 and q6000 jobs so the three logs are directly
76|# comparable. Same order, same flags -- a different order would invite a
77|# comparison-of-typos.
78|run "len=256,  smax=64,  8000 pairs, ident=90%"  --len 256  --pairs 8000 --ident 90 --smax 64  --reps 20
79|run "len=1024, smax=64,  2000 pairs, ident=90%"  --len 1024 --pairs 2000 --ident 90 --smax 64  --reps 20
80|run "len=1024, smax=256, 2000 pairs, ident=90%"  --len 1024 --pairs 2000 --ident 90 --smax 256 --reps 20
81|run "len=1024, smax=256, 2000 pairs, ident=70%"  --len 1024 --pairs 2000 --ident 70 --smax 256 --reps 20
82|
83|echo
84|echo "=== H6-V100 RESULT ==="
85|echo "  regimes verified and timed : $rc_verified"
86|if [ "$rc_all" -eq 0 ]; then echo "H6-V100: PASS"; else echo "H6-V100: FAIL"; fi
87|exit "$rc_all"
88|