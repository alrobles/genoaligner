#!/bin/bash
# End-to-end validation of the three-oracle Fase 3 path, on the cluster, no GPU.
# Proves: (a) correct input passes, (b) a GPU error is caught, (c) a SeqAn3
# disagreement is caught, (d) EMPTY input is NOT reported as a pass.
set -u
REPO=/beegfs/a474r867/genoaligner/repo
W=/beegfs/a474r867/genoaligner/seqan_probe
cd "$W" || exit 1

# 1. Build the emit file from the deterministic control set (acts as the GPU output).
python3 - <<'PYEOF'
rows = []
with open('/beegfs/a474r867/genoaligner/repo/tests/parity/data/control_seed12345.tsv') as f:
    for line in f:
        if line.startswith('#') or line.startswith('pattern'):
            continue
        p = line.rstrip('\n').split('\t')
        if len(p) < 5:
            continue
        rows.append(p)
        if len(rows) >= 300:
            break
with open('/beegfs/a474r867/genoaligner/seqan_probe/synth_emit.tsv', 'w') as o:
    o.write('index\tpattern\ttext\tgpu\tcpu\tlabel\n')
    for i, p in enumerate(rows):
        o.write(f'{i}\t{p[0]}\t{p[1]}\t0\t0\t{p[4]}\n')
print('emit cases:', len(rows))
PYEOF

# 2. SeqAn3 oracle adds column 6.
LD_LIBRARY_PATH=/kuhpc/sw/gcc/14.2/lib64 ./seqan3_oracle synth_emit.tsv seqan_out.tsv

# 3. Case A: simulate a fully CORRECT GPU by computing the true edit distance
#    (edlib via the oracle) and writing it into the gpu column. Simply copying
#    the seqan3 column would be wrong: seqan3 differs from the truth whenever the
#    GPU is right by coincidence, so the "correct" file must carry the real value.
python3 - <<'PYEOF'
import sys
sys.path.insert(0, '/beegfs/a474r867/genoaligner/repo/tests/parity')
import os
os.chdir('/beegfs/a474r867/genoaligner/repo')
import importlib.util
spec = importlib.util.spec_from_file_location(
    "oracle", "/beegfs/a474r867/genoaligner/repo/tests/parity/oracle_external.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)

src = '/beegfs/a474r867/genoaligner/seqan_probe/seqan_out.tsv'
dst = '/beegfs/a474r867/genoaligner/seqan_probe/emit_correct.tsv'
with open(src) as f, open(dst, 'w') as o:
    for line in f:
        p = line.rstrip('\n').split('\t')
        if p[0] == 'index':
            o.write(line)
            continue
        if len(p) < 6:
            continue
        p[3] = str(oracle.both(p[1], p[2]))   # the TRUE edit distance
        o.write('\t'.join(p) + '\n')
print('emit_correct.tsv written with true edit distances')
PYEOF

# 4. Case B: inject the R1 bug on one row -> expect FAIL
awk -F'\t' 'BEGIN{OFS="\t"} NR==1{print;next} NF>=6{ if ($1=="5") $4=$6+1; print }' \
    seqan_out.tsv > emit_bug.tsv

# 5. Case C: corrupt the seqan3 column on one row -> expect FAIL (seqan3 disagreement)
awk -F'\t' 'BEGIN{OFS="\t"} NR==1{print;next} NF>=6{ if ($1=="7") $6=$6+1; print }' \
    seqan_out.tsv > emit_seqanbad.tsv

# 6. Case D: empty file -> expect INCONCLUSIVE (exit 2), NOT pass
printf 'index\tpattern\ttext\tgpu\tcpu\tseqan3\tlabel\n' > emit_empty.tsv

run_case() {
    local label="$1" file="$2" expect="$3"
    cd "$REPO" || exit 1
    local out rc
    out=$(python3 tests/parity/oracle_external.py --compare "$file" 2>&1)
    rc=$?
    local verdict
    if [ "$rc" -eq "$expect" ]; then verdict="OK"; else verdict="** UNEXPECTED **"; fi
    printf '%-28s exit=%d expect=%d  %s\n' "$label" "$rc" "$expect" "$verdict"
    echo "$out" | grep -E 'parity|seqan3 |Fase 3' | sed 's/^/    /'
}

echo
echo "=== Fase 3 oracle end-to-end validation ==="
run_case "A correct gpu"       "$W/emit_correct.tsv"  0
run_case "B R1-shaped bug"     "$W/emit_bug.tsv"      1
run_case "C seqan3 corrupted"  "$W/emit_seqanbad.tsv" 1
run_case "D empty input"       "$W/emit_empty.tsv"    2
