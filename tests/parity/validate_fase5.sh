#!/bin/bash
# Fase 5 validation: the portability layer must build BOTH backends from one
# source, and must refuse a mismatched compiler.
#
# Run on the KU login node (no GPU needed: the probe is a host-side program that
# reports which backend it was compiled for).
set -u

export LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64:${LD_LIBRARY_PATH:-}
CMAKE=/kuhpc/sw/cmake/3.30.3/gcc/14.2/bin/cmake
ROCM=/kuhpc/sw/rocm/6.4.3
CUDA=/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4
ROOT=/beegfs/a474r867/genoaligner/repo
cd "$ROOT" || exit 1

source /kuhpc/sw/lmod/9.3/init/bash 2>/dev/null
ml purge >/dev/null 2>&1
ml load compiler/gcc/14.2 >/dev/null 2>&1   # BEFORE cmake: GLIBCXX_3.4.32
ml load rocm/6.4.3 >/dev/null 2>&1

fails=0
report() {  # label, expected_rc, actual_rc
    if [ "$2" -eq "$3" ]; then
        printf '  %-42s exit=%d  OK\n' "$1" "$3"
    else
        printf '  %-42s exit=%d  ** EXPECTED %d **\n' "$1" "$3" "$2"
        fails=$((fails+1))
    fi
}

echo "=== Fase 5: portability layer validation ==="

# --- TEST 1: mismatched compiler must be refused ----------------------------
echo
echo "--- T1: backend=cuda with hipcc (must be refused) ---"
rm -rf /tmp/g5_bad
$CMAKE -S . -B /tmp/g5_bad -DGENOALIGNER_BACKEND=cuda \
       -DCMAKE_CXX_COMPILER=$ROCM/bin/hipcc >/tmp/g5_t1.log 2>&1
rc=$?
report "T1 refused mismatched compiler" 1 "$rc"
grep -q "needs CMAKE_CXX_COMPILER=nvcc" /tmp/g5_t1.log \
    && echo "     message names the right compiler: yes" \
    || { echo "     ** message missing **"; fails=$((fails+1)); }

# --- TEST 2: ROCm backend configures AND builds -----------------------------
echo
echo "--- T2: backend=rocm configure + build ---"
rm -rf /tmp/g5_rocm
$CMAKE -S . -B /tmp/g5_rocm -DGENOALIGNER_BACKEND=rocm \
       -DCMAKE_CXX_COMPILER=$ROCM/bin/hipcc \
       -DCMAKE_BUILD_TYPE=Release >/tmp/g5_t2.log 2>&1
rc=$?
report "T2a rocm configure" 0 "$rc"
if [ "$rc" -eq 0 ]; then
    $CMAKE --build /tmp/g5_rocm -j 8 >>/tmp/g5_t2.log 2>&1
    rc2=$?
    report "T2b rocm build" 0 "$rc2"
    if [ "$rc2" -eq 0 ]; then
        echo "     probe output:"
        /tmp/g5_rocm/genoaligner_probe 2>&1 | sed 's/^/       /' | head -8
    fi
else
    tail -12 /tmp/g5_t2.log | sed 's/^/     /'
fi

# --- TEST 3: CUDA backend configures AND builds -----------------------------
echo
echo "--- T3: backend=cuda configure + build (nvcc) ---"
rm -rf /tmp/g5_cuda
$CMAKE -S . -B /tmp/g5_cuda -DGENOALIGNER_BACKEND=cuda \
       -DCMAKE_CXX_COMPILER=$CUDA/bin/nvcc \
       -DGENOALIGNER_CUDA_ROOT=$CUDA \
       -DCMAKE_CXX_FLAGS="-w -gencode arch=compute_75,code=sm_75" \
       -DCMAKE_BUILD_TYPE=Release >/tmp/g5_t3.log 2>&1
rc=$?
report "T3a cuda configure" 0 "$rc"
if [ "$rc" -eq 0 ]; then
    $CMAKE --build /tmp/g5_cuda -j 8 >>/tmp/g5_t3.log 2>&1
    rc2=$?
    report "T3b cuda build" 0 "$rc2"
    if [ "$rc2" -eq 0 ]; then
        echo "     embedded GPU code:"
        $CUDA/bin/cuobjdump --list-elf /tmp/g5_cuda/genoaligner_probe 2>/dev/null \
            | sed 's/^/       /' | head -4
        echo "     probe output:"
        /tmp/g5_cuda/genoaligner_probe 2>&1 | sed 's/^/       /' | head -8
    fi
else
    tail -12 /tmp/g5_t3.log | sed 's/^/     /'
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "=== FASE 5 VALIDATION: PASS ==="
else
    echo "=== FASE 5 VALIDATION: $fails FAILURE(S) ==="
fi
exit "$fails"
