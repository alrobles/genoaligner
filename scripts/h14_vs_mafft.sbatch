#!/bin/bash -l
#SBATCH --job-name=h14_vs_mafft
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/h14_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/h14_%j.err

# H14: genoaligner vs MAFFT on REAL pipeline data, on the task they can both do.
#
# The task, declared up front: "align every sequence in a gene file to a
# reference, pairwise" -- the shape orthology/RBH actually needs. genoaligner
# is a PAIRWISE aligner; MAFFT is an MSA aligner used here in its per-pair
# mode (mafft --auto on 2-sequence inputs), which is exactly how the pipeline
# invokes it (scripts/align_genes_array.sbatch: mafft --auto --thread N).
# This is NOT a claim that genoaligner replaces MAFFT for MSAs.
#
# What is measured, all in ONE job on ONE node:
#   A) genoaligner align_sw_batch WITH CIGAR on all N-1 pairs (capability)
#   B) genoaligner score-only on the same pairs (throughput ceiling)
#   C) MAFFT --auto, one process per pair, serial (the naive pipeline shape)
#   D) MAFFT --auto, 8 concurrent processes (the realistic CPU ceiling)
#   E) SeqAn3 scores every genoaligner result -- external verification on real data
set -o pipefail

GENE=${GENE:-CYTB}
IN=/beegfs/a474r867/phylogenyAI/data/genes_final/${GENE}.fasta
ROOT=/beegfs/a474r867/genoaligner/repo
LOGD=/beegfs/a474r867/genoaligner/logs
PROBE=/beegfs/a474r867/genoaligner/seqan_probe
D=/beegfs/a474r867/genoaligner/vs_mafft_${SLURM_JOB_ID}
mkdir -p "$D/pairs"
cd "$ROOT" || exit 1

source /beegfs/a474r867/miniconda3/etc/profile.d/conda.sh
conda activate genoml-phylo
echo "=== host: $(hostname) ==="
mafft --version 2>&1 | head -1
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
rocm-smi --showpids 2>/dev/null || true
echo "input: $IN ($(grep -c '^>' "$IN") seqs)"

# --- build the driver (hipcc; fasta reader is header-only) -------------------
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -pthread -I. -Iinclude \
    -o "$D/vs_ref" bench/vs_reference.cpp src/api/api.cpp || { echo "BUILD FAILED"; exit 1; }

# --- split the gene file into per-pair 2-seq files for mafft ------------------
# Record 0 is the reference (declared); every other record is a pair partner.
python3 - "$IN" "$D/pairs" <<'PY'
import sys, os
src, outdir = sys.argv[1], sys.argv[2]
recs, cur, name = [], [], None
for line in open(src):
    if line.startswith('>'):
        if name is not None: recs.append((name, ''.join(cur)))
        name, cur = line[1:].split()[0], []
    else:
        cur.append(line.strip())
if name is not None: recs.append((name, ''.join(cur)))
ref_name, ref_seq = recs[0]
open(os.path.join(outdir, 'ref.fa'), 'w').write(f'>{ref_name}\n{ref_seq}\n')
with open(os.path.join(outdir, 'order.txt'), 'w') as order:
    for i, (n, s) in enumerate(recs[1:], 1):
        with open(os.path.join(outdir, f'{i:05d}.fa'), 'w') as f:
            f.write(f'>{ref_name}\n{ref_seq}\n>{n}\n{s}\n')
        order.write(f'{i}\n')
print(f'pairs: {len(recs)-1}, ref={ref_name} ({len(ref_seq)} bp)')
PY

RC=0

# --- A/B: genoaligner, with CIGAR and score-only ------------------------------
echo; echo "=== genoaligner: ALL pairs, WITH CIGAR (capability) ==="
"$D/vs_ref" "$IN" --emit "$D/geno_cigar.tsv" --reps 5 || RC=1
echo; echo "=== genoaligner: ALL pairs, score-only (throughput) ==="
"$D/vs_ref" "$IN" --no-cigar --emit "$D/geno_scoreonly.tsv" --reps 7 || RC=1

# --- C: MAFFT serial, one process per pair -----------------------------------
# Time-boxed: if serial cannot finish inside the job we report the measured
# rate over completed pairs, NOT an extrapolated total.
echo; echo "=== mafft --auto, serial (timeout 40 min) ==="
T0=$SECONDS
timeout 2400 sh -c 'while read -r i; do
    f=$(printf "%05d" "$i")
    mafft --auto --thread 1 --quiet "'$D'/pairs/$f.fa" > "'$D'/pairs/$f.aln" 2>/dev/null
done < "'$D'/pairs/order.txt"'
MAFFT_SERIAL_S=$((SECONDS - T0))
MAFFT_DONE=$(ls "$D"/pairs/*.aln 2>/dev/null | wc -l)
NPAIRS=$(wc -l < "$D/pairs/order.txt")
echo "  serial wall: ${MAFFT_SERIAL_S} s; completed ${MAFFT_DONE}/${NPAIRS} pairs"

# --- D: MAFFT, 8 concurrent processes ----------------------------------------
echo; echo "=== mafft --auto, 8 concurrent pairs ==="
T0=$SECONDS
while read -r i; do printf "%05d\n" "$i"; done < "$D/pairs/order.txt" | \
xargs -P 8 -I{} sh -c \
    'mafft --auto --thread 1 --quiet "'$D'/pairs/{}.fa" > "'$D'/pairs/{}.p8.aln" 2>/dev/null'
MAFFT_PAR_S=$((SECONDS - T0))
echo "  8-way wall: ${MAFFT_PAR_S} s for $NPAIRS pairs"

# --- E: SeqAn3 verifies genoaligner's scores on the real data ----------------
echo; echo "=== SeqAn3 on genoaligner's real-data scores ==="
LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64 "$PROBE/seqan3_sw_oracle" --selftest || RC=1
LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64 \
    "$PROBE/seqan3_sw_oracle" "$D/geno_cigar.tsv" "$D/oracle_cigar.tsv" || RC=1

# --- F: do the alignments agree? per-pair identity comparison -----------------
# genoaligner (local) vs mafft (global): objectives differ legitimately, so we
# compare identities, not equality. Large systematic divergence = investigate.
echo; echo "=== identity comparison: genoaligner CIGAR vs mafft alignment ==="
python3 - "$D" "$D/pairs/order.txt" <<'PY'
import sys, re, os
D, orderf = sys.argv[1], sys.argv[2]
# genoaligner results, keyed by source record index == pair file index
geno = {}
for line in open(os.path.join(D, 'geno_cigar.tsv')):
    if line.startswith('idx'): continue
    f = line.rstrip('\n').split('\t')
    geno[int(f[0])] = f   # idx text pattern score si sj ei ej cigar
def aln_pair(path):
    seqs, cur = [], []
    for line in open(path):
        if line.startswith('>'):
            if cur: seqs.append(''.join(cur)); cur = []
        else: cur.append(line.strip())
    if cur: seqs.append(''.join(cur))
    return seqs
n = both = 0
diffs, worst = [], []
for i in [int(x) for x in open(orderf)]:
    ap = os.path.join(D, 'pairs', f'{i:05d}.aln')
    if not os.path.exists(ap) or i not in geno: continue
    n += 1
    s = aln_pair(ap)
    if len(s) != 2: continue
    both += 1
    a, b = s
    same = sum(1 for x, y in zip(a, b) if x == y and x != '-')
    mafft_id = same / len(a) if a else 0
    # genoaligner local identity = M ops / aligned ops
    ops = re.findall(r'(\d+)([MXID])', geno[i][8])
    m = sum(int(c) for c, o in ops if o == 'M')
    tot = sum(int(c) for c, o in ops)
    geno_id = m / tot if tot else (1.0 if int(geno[i][3]) == 0 else 0)
    diffs.append(abs(mafft_id - geno_id))
    worst.append((abs(mafft_id - geno_id), i, mafft_id, geno_id))
if both:
    diffs.sort()
    med = diffs[len(diffs)//2]
    big = sum(1 for d in diffs if d > 0.10)
    print(f'compared={both}  median|Δidentity|={med:.3f}  pairs Δ>0.10={big}')
    for d, i, mi, gi in sorted(worst, reverse=True)[:8]:
        print(f'  pair {i}: mafft_id={mi:.3f} geno_id={gi:.3f}')
else:
    print('no overlapping results to compare')
PY

echo
echo "=== H14 RESULT (gene=$GENE) ==="
echo "mafft_serial_s=$MAFFT_SERIAL_S  mafft_8way_s=$MAFFT_PAR_S"
[ "$RC" -eq 0 ] && echo "H14: PASS" || echo "H14: FAIL"
exit "$RC"
