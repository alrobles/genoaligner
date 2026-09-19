#!/bin/bash
#SBATCH --job-name=genomsa_lf
#SBATCH --partition=sixhour
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#
# genomsa --codon --local-frame over the 31-locus QC'd gene panel.
# CPU-shim binary (bit-identical to GPU path; HIP runtime not needed).
# Array task = gene index into the fixed 31-gene list.
#
# Inputs : $PAI/data/genes_qc_pass/<GENE>.fasta   (same as the baseline run)
# Outputs: $OUTDIR/<GENE>.codon.fasta + .qc.tsv + .log
set -u
PAI=/beegfs/a474r867/phylogenyAI
REPO=/beegfs/a474r867/genoaligner/repo
OUTDIR=${LF_OUT:-$PAI/data/genomsa_codon_lf}
BIN=$OUTDIR/bin
mkdir -p "$OUTDIR" "$BIN"

# atomic CPU-shim build (same recipe as codon_smoke_cpu.sbatch)
if [ ! -x "$BIN/genomsa_cpu" ]; then
    (
        flock -x 9
        if [ ! -x "$BIN/genomsa_cpu" ]; then
            g++ -O2 -std=c++17 -DGENOALIGNER_HIP_SHIM \
                -I"$REPO/tests/parity/hip_cpu_shim" -I"$REPO/include" \
                -pthread "$REPO/tools/genomsa.cpp" \
                "$REPO/src/msa/msa_gpu.cpp" "$REPO/src/msa/msa_ref.cpp" \
                "$REPO/src/msa/nj_gpu.cpp" \
                -o "$BIN/genomsa_cpu.tmp.$$" 2>>"$OUTDIR/build_cpu.log" \
                && mv "$BIN/genomsa_cpu.tmp.$$" "$BIN/genomsa_cpu"
        fi
    ) 9>"$OUTDIR/build.lock"
fi

GENES=(A2AB ADORA3 ADRB2 APOB APP ATP7 BCHE BDNF BMI1 BRCA1 BRCA2 CNR1 COI
       CREM CYTB DMP1 EDG1 ENAM FBN1 GHR IRBP ND1 ND2 PLCB4 PNOC RAG1A
       RAG1B RAG2 TTN TYR1 VWF)
GENE=${GENES[${SLURM_ARRAY_TASK_ID:-0}]}
# Mitochondrial loci use the vertebrate mito code (same split as
# align_mt_qc_array.sbatch: macse -gc_def 2). Under gc=1, TGA (=Trp in
# mito) reads as stop and --local-frame fires spurious fs events.
GC=1
case "$GENE" in COI|CYTB|ND1|ND2) GC=2;; esac
IN=$PAI/data/genes_qc_pass/$GENE.fasta
OUT=$OUTDIR/$GENE.codon.fasta

{ "$BIN/genomsa_cpu" "$IN" "$OUT" --codon --gc-def $GC --cpu --local-frame \
      --codon-qc "$OUTDIR/$GENE.qc.tsv" \
  && echo "LF_OK $GENE"; } > "$OUTDIR/$GENE.log" 2>&1
