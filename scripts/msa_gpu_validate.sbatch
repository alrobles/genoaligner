#!/bin/bash -l
#SBATCH --job-name=msa_gpu_val
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/msaval_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/msaval_%j.err

# genomsa on-device validation (H-MSA-1).
#
#   1. build tools/genomsa with hipcc (gfx90a default on the node)
#   2. device-vs-CPU-reference parity: first $SUBN records of $GENE run
#      through BOTH paths -- must be byte-identical (same algorithm; any
#      difference is a device bug, not a "close enough")
#   3. full-gene GPU run + timing for $GENE and $GENE2
#   4. sanity stats vs the MACSE trimmed alignment when one exists
#      (different algorithm -> statistics, not byte equality)
#
# GENE/BIG are overridable:  sbatch --export=GENE=ADORA3,BIG=COI ...
set -o pipefail

PAI=/beegfs/a474r867/phylogenyAI
REPO=/beegfs/a474r867/genoaligner/repo
OUT=${OUT_DIR:-$PAI/data/msa_genomsa}
GENE=${GENE:-BMI1}
BIG=${BIG:-COI}
SUBN=${SUBN:-150}
mkdir -p "$OUT"
cd "$REPO" || exit 1

echo "=== host: $(hostname)  $(date) ==="
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. -Iinclude \
    -o /tmp/genomsa tools/genomsa.cpp src/msa/msa_gpu.cpp src/msa/msa_ref.cpp \
    || { echo "BUILD FAILED"; exit 1; }
echo "build ok"

FA="$PAI/data/genes_qc_pass/${GENE}.fasta"
FB="$PAI/data/genes_qc_pass/${BIG}.fasta"
[ -s "$FA" ] || { echo "missing $FA"; exit 1; }
[ -s "$FB" ] || { echo "missing $FB"; exit 1; }

RC=0

# ---- 2. device parity on a subset (must be BYTE-IDENTICAL) --------------
head -n $((SUBN * 2)) "$FA" > /tmp/sub_${GENE}.fasta   # 2 lines per record
echo; echo "### parity subset: $GENE first $SUBN records ###"
/tmp/genomsa /tmp/sub_${GENE}.fasta /tmp/sub_gpu.aln 2>/tmp/g1.log || { echo "GPU run failed"; cat /tmp/g1.log; exit 1; }
cat /tmp/g1.log
/tmp/genomsa /tmp/sub_${GENE}.fasta /tmp/sub_cpu.aln --cpu 2>/tmp/c1.log || { echo "CPU run failed"; cat /tmp/c1.log; exit 1; }
cat /tmp/c1.log
if diff -q /tmp/sub_gpu.aln /tmp/sub_cpu.aln >/dev/null; then
    echo "PARITY: PASS (GPU == CPU-ref, byte-identical, on-device)"
else
    echo "PARITY: FAIL -- device output differs from reference"; RC=1
    diff /tmp/sub_gpu.aln /tmp/sub_cpu.aln | head -10
fi

# ---- 3. full-gene GPU runs ----------------------------------------------
echo; echo "### full $GENE (GPU) ###"
/tmp/genomsa "$FA" "$OUT/${GENE}.genomsa.aln.fasta" 2>&1 | tail -4
echo; echo "### full $BIG (GPU) ###"
time /tmp/genomsa "$FB" "$OUT/${BIG}.genomsa.aln.fasta" 2>&1 | tail -4

# ---- 4. sanity stats vs MACSE trimmed alignment (if present) ------------
python3 - "$OUT" "$PAI" "$GENE" "$BIG" <<'PY'
import sys, os
out, pai = sys.argv[1], sys.argv[2]
def load(p):
    seqs={}; cur=None
    for line in open(p):
        line=line.strip()
        if line.startswith('>'): cur=line[1:].split()[0]; seqs[cur]=''
        elif cur: seqs[cur]+=line
    return seqs
for gene in sys.argv[3:]:
    g = load(f"{out}/{gene}.genomsa.aln.fasta")
    w = len(next(iter(g.values())))
    ragged = sum(1 for s in g.values() if len(s)!=w)
    gc = sum(c=='-' for s in g.values() for c in s)
    tot = sum(len(s) for s in g.values())
    print(f"{gene}: n={len(g)} width={w} ragged={ragged} gaps={100*gc/tot:.1f}%")
    ref = f"{pai}/data/alignments_qc_trimmed/{gene}.trimmed.aln.fasta"
    if os.path.exists(ref):
        m = load(ref)
        wm = len(next(iter(m.values())))
        ids = set(g) & set(m)
        print(f"  vs MACSE-trimmed: width={wm} shared_ids={len(ids)}")
    # per-seq letter preservation (no corruption): ungap == input
    raw = load(f"{pai}/data/genes_qc_pass/{gene}.fasta")
    bad = sum(1 for k,s in g.items() if k in raw and s.replace('-','') != raw[k])
    print(f"  ungap==input violations: {bad}")
PY

echo
[ "$RC" -eq 0 ] && echo "MSA-VAL: PASS" || echo "MSA-VAL: FAIL"
exit "$RC"
