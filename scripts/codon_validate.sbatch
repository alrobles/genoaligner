#!/bin/bash -l
#SBATCH --job-name=codon_val
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/codon_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/codon_%j.err

# Codon-mode validation (alpha=65).
#
#   1. build genomsa + test_codon with the node toolchain (hipcc/nvcc)
#   2. codon unit tests on the node
#   3. per gene in $GENES: --codon --cpu vs --codon (GPU) must be
#      BYTE-IDENTICAL; decoded rows must ungap to the encoded content;
#      alignment length must be a multiple of 3 (whole-codon indels)
#   4. per-gene walltime cpu vs gpu -> $OUT/codon_<job>_<backend>.tsv
#
# Overrides: sbatch --export=REPO=...,GENES="A2AB COI CYTB" scripts/codon_validate.sbatch
# NVIDIA: sbatch --gres=gpu:a100:1 --export=ALL,BACKEND=cuda,CUDA_ARCH=sm_80
set -o pipefail

PAI=/beegfs/a474r867/phylogenyAI
REPO=${REPO:-/beegfs/a474r867/genoaligner/repo}
GENES_DIR=${GENES_DIR:-$PAI/data/genes_qc_pass}
OUT=${OUT_DIR:-$PAI/data/codon_val}
GENES=${GENES:-"A2AB CNR1 COI CYTB"}
ROCM=${ROCM:-/kuhpc/sw/rocm/6.4.3}
HIPCC=${HIPCC:-$ROCM/bin/hipcc}
BACKEND=${BACKEND:-rocm}
CUDA=${CUDA:-/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4}
CUDA_ARCH=${CUDA_ARCH:-sm_80}
mkdir -p "$OUT"
cd "$REPO" || exit 1

echo "=== host: $(hostname)  $(date)  job ${SLURM_JOB_ID:-local}  backend $BACKEND ==="
echo "=== commit: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD) ==="

if [ "$BACKEND" = cuda ]; then
    nvidia-smi -L 2>/dev/null | head -1
    build() { out=$1; shift
        "$CUDA/bin/nvcc" -O2 -w -std=c++17 -D__HIP_PLATFORM_NVIDIA__ \
            -gencode "arch=compute_${CUDA_ARCH#sm_},code=$CUDA_ARCH" \
            -I "$ROCM/include" -I "$CUDA/include" -I "$CUDA/targets/x86_64-linux/include" \
            -I. -Iinclude -lcuda -L "$CUDA/lib64/stubs" -x cu -o "$out" "$@"; }
else
    rocm-smi --showproductname 2>/dev/null | grep -i "card series" | head -1
    build() { out=$1; shift
        "$HIPCC" -O2 -ffp-contract=off -std=c++17 -I. -Iinclude -o "$out" "$@"; }
fi

BIN=/tmp/codon_${SLURM_JOB_ID:-local}
mkdir -p "$BIN"
build "$BIN/genomsa" tools/genomsa.cpp src/msa/msa_gpu.cpp src/msa/nj_gpu.cpp \
    src/msa/msa_ref.cpp src/io/fasta.cpp || { echo "BUILD FAILED (genomsa)"; exit 1; }
g++ -O2 -std=c++17 -Iinclude -o "$BIN/test_codon" \
    src/msa/msa_ref.cpp tests/msa/test_codon.cpp -lpthread \
    || { echo "BUILD FAILED (test_codon)"; exit 1; }
echo "build ok"

RC=0
echo; echo "### codon unit tests ###"
"$BIN/test_codon" || RC=1

# mito genes use NCBI genetic code 2
gc_of() { case "$1" in COI|CYTB|ND1|ND2) echo 2;; *) echo 1;; esac; }

TSV="$OUT/codon_${SLURM_JOB_ID:-local}_${BACKEND}.tsv"
printf "gene\tnseq\tgc\taln_len\tpct_gap\tstops\tpartials\tframe1\tframe2\tt_cpu\tt_gpu\tparity\n" > "$TSV"

for G in $GENES; do
    FA="$GENES_DIR/$G.fasta"
    [ -s "$FA" ] || { echo "SKIP $G (no fasta)"; continue; }
    GC=$(gc_of "$G")
    echo; echo "### $G (gc=$GC) ###"

    t0=$(date +%s.%N)
    "$BIN/genomsa" "$FA" "$BIN/$G.cpu.fasta" --codon --gc-def "$GC" --cpu \
        --codon-qc "$BIN/$G.qc.tsv" 2>"$BIN/$G.cpu.log" || { echo "CPU FAIL $G"; tail -3 "$BIN/$G.cpu.log"; RC=1; continue; }
    t1=$(date +%s.%N)
    "$BIN/genomsa" "$FA" "$BIN/$G.gpu.fasta" --codon --gc-def "$GC" \
        2>"$BIN/$G.gpu.log" || { echo "GPU FAIL $G"; tail -3 "$BIN/$G.gpu.log"; RC=1; continue; }
    t2=$(date +%s.%N)
    TCPU=$(echo "$t1 - $t0" | bc); TGPU=$(echo "$t2 - $t1" | bc)
    tail -2 "$BIN/$G.cpu.log"; tail -2 "$BIN/$G.gpu.log"

    PARITY=FAIL
    cmp -s "$BIN/$G.cpu.fasta" "$BIN/$G.gpu.fasta" && PARITY=PASS
    [ "$PARITY" = PASS ] || RC=1
    echo "parity: $PARITY"

    # biological invariants + stats (python3 on the node)
    python3 - "$FA" "$BIN/$G.cpu.fasta" "$BIN/$G.qc.tsv" "$G" "$GC" \
            "$TCPU" "$TGPU" "$PARITY" >> "$TSV" <<'PY'
import sys
fa, aln_f, qc_f, gene, gc, tc, tg, par = sys.argv[1:9]
def read_fa(p):
    recs, name, buf = [], None, []
    for ln in open(p):
        ln = ln.strip()
        if ln.startswith('>'):
            if name is not None: recs.append((''.join(buf)))
            name, buf = ln[1:], []
        else: buf.append(ln)
    if name is not None: recs.append(''.join(buf))
    return recs
raw = read_fa(fa); aln = read_fa(aln_f)
assert len(raw) == len(aln), "row count mismatch"
ncol = len(aln[0])
ok = all(len(r) == ncol for r in aln) and ncol % 3 == 0
frame_viol = 0
for r in aln:
    for i in range(0, ncol, 3):
        col = r[i:i+3]
        if '-' in col and col != '---': frame_viol += 1
stops = partials = f1 = f2 = 0
for ln in open(qc_f):
    _, fr, st, pt = ln.strip().split('\t')
    stops += int(st); partials += int(pt)
    f1 += fr == '1'; f2 += fr == '2'
pct_gap = 100.0 * sum(r.count('-') for r in aln) / (len(aln) * ncol)
print(f"{gene}\t{len(raw)}\t{gc}\t{ncol}\t{pct_gap:.1f}\t{stops}\t"
      f"{partials}\t{f1}\t{f2}\t{float(tc):.1f}\t{float(tg):.1f}\t{par}"
      + ("" if ok and frame_viol == 0 else "\tINVARIANT_FAIL"))
if not ok: print(f"# {gene}: ragged or len%3!=0", file=sys.stderr)
if frame_viol: print(f"# {gene}: {frame_viol} split-codon gaps", file=sys.stderr)
PY
done

echo; echo "=== codon_${SLURM_JOB_ID:-local}_${BACKEND}.tsv ==="
column -t "$TSV"
NPASS=$(awk -F'\t' 'NR>1 && $12=="PASS"' "$TSV" | wc -l)
NTOT=$(awk -F'\t' 'NR>1' "$TSV" | wc -l)
if [ "$NPASS" -eq "$NTOT" ] && [ "$NTOT" -gt 0 ] && [ "$RC" -eq 0 ]; then
    echo "CODON VALIDATION: PASS ($NPASS genes byte-identical cpu/gpu)"
else
    echo "CODON VALIDATION: FAIL"; RC=1
fi
exit $RC
