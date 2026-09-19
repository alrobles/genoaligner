#!/bin/bash -l
#SBATCH --job-name=genomsa_31
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/genomsa31_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/genomsa31_%j.err

# genomsa (wavefront kernel) on all 31 QC-passed genes + stats vs MACSE.
# Output: $PAI/data/msa_genomsa/<GENE>.genomsa.aln.fasta
#         $PAI/data/msa_genomsa/compare.tsv  (per-gene stats + SPS/SP-cost)
set -o pipefail

PAI=/beegfs/a474r867/phylogenyAI
REPO=/beegfs/a474r867/genoaligner/repo
OUT=$PAI/data/msa_genomsa
mkdir -p "$OUT"
cd "$REPO" || exit 1

echo "=== host: $(hostname)  $(date) ==="
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -ffp-contract=off -std=c++17 -I. -Iinclude \
    -o /tmp/genomsa tools/genomsa.cpp src/msa/msa_gpu.cpp src/msa/msa_ref.cpp \
    || { echo "BUILD FAILED"; exit 1; }
echo "build ok"

echo -e "gene\tn\twidth\tgaps\tmacse_w\tsps\tsp_genomsa\tsp_macse" > "$OUT/compare.tsv"
: > /tmp/timings.tsv
FAILED=""

for FA in "$PAI"/data/genes_qc_pass/*.fasta; do
    GENE=$(basename "$FA" .fasta)
    echo; echo "### $GENE ###"
    /tmp/genomsa "$FA" "$OUT/${GENE}.genomsa.aln.fasta" 2>/tmp/${GENE}.log \
        || { FAILED="$FAILED $GENE"; continue; }
    tail -4 /tmp/${GENE}.log
    # 'levels=.. pairs=.. dir_peak=..MB | dist=.. tree=.. align=..' / 'align total: ..s'
    L=$(grep -o 'dist=[0-9.]*s tree=[0-9.]*s align=[0-9.]*s' /tmp/${GENE}.log | tail -1)
    T=$(grep -o 'align total: [0-9.]*s' /tmp/${GENE}.log | tail -1)
    echo -e "$GENE\t$L $T" >> /tmp/timings.tsv
done
cat /tmp/timings.tsv

# ---- comparison vs MACSE where available ----------------------------------
python3 - "$OUT" "$PAI" <<'PY'
import sys, os, itertools
out, pai = sys.argv[1], sys.argv[2]
def load(p):
    seqs={}; cur=None
    for line in open(p):
        line=line.strip()
        if line.startswith(">"): cur=line[1:].split()[0]; seqs[cur]=""
        elif cur: seqs[cur]+=line
    return seqs
def posmap(aln):
    out=[]; r=0
    for c in aln:
        out.append(r if c!="-" else -1)
        if c!="-": r+=1
    return out
def sps(g, m, ids):  # fraction of genomsa aligned residue-pairs also aligned in MACSE
    pm={i:posmap(g[i]) for i in ids}; qm={i:posmap(m[i]) for i in ids}
    same=tot=0
    for a,b in itertools.combinations(ids,2):
        mp={}
        qa,qb=qm[a],qm[b]
        for j in range(len(qa)):
            if qa[j]>=0 and qb[j]>=0: mp[(qa[j],qb[j])]=1
        pa,pb=pm[a],pm[b]
        for j in range(len(pa)):
            if pa[j]>=0 and pb[j]>=0:
                tot+=1
                if (pa[j],pb[j]) in mp: same+=1
    return (same/tot) if tot else float("nan")
def sp_cost(aln, ids):
    w=len(aln[ids[0]]); c=0
    for j in range(w):
        col=[aln[i][j] for i in ids]
        for a,b in itertools.combinations(col,2):
            if a=="-" and b=="-": continue
            if a=="-" or b=="-" or a!=b: c+=1
    return c
rows=[]
for f in sorted(os.listdir(out)):
    if not f.endswith(".genomsa.aln.fasta"): continue
    gene=f.split(".")[0]
    g=load(f"{out}/{f}")
    w=len(next(iter(g.values())))
    gc=sum(c=="-" for s in g.values() for c in s)
    tot=sum(len(s) for s in g.values())
    mf=f"{pai}/data/alignments_qc/{gene}.aln.fasta"
    mw=sps_s=sg=sm=""
    if os.path.exists(mf):
        m=load(mf); mw=len(next(iter(m.values())))
        ids=sorted(set(g)&set(m))
        if len(ids)>200:                    # SPS is O(w*n^2): deterministic subsample
            step=len(ids)/200.0
            ids=[ids[int(k*step)] for k in range(200)]
        if len(ids)>=2:
            sps_v=sps(g,m,ids); sps_s=f"{sps_v:.3f}"
            sg=sp_cost(g,ids); sm=sp_cost(m,ids)
    rows.append((gene,len(g),w,f"{100*gc/tot:.1f}",mw,sps_s,sg,sm))
tim={}
if os.path.exists("/tmp/timings.tsv"):
    for line in open("/tmp/timings.tsv"):
        p=line.split("\t",1)
        if len(p)==2: tim[p[0]]=p[1].strip()
with open(f"{out}/timings.tsv","w") as fh:
    for k,v in tim.items(): fh.write(f"{k}\t{v}\n")
with open(f"{out}/compare.tsv","a") as fh:
    for r in rows:
        print("\t".join(str(x) for x in r))
        fh.write("\t".join(str(x) for x in r)+"\n")
PY

echo
[ -z "$FAILED" ] && echo "GENOMSA-31: PASS" || echo "GENOMSA-31: FAIL ($FAILED)"
