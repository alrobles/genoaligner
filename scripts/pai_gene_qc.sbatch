#!/bin/bash -l
#SBATCH --job-name=pai_gene_qc
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/genoaligner/logs/paiqc_%j.out
#SBATCH --error=/beegfs/a474r867/genoaligner/logs/paiqc_%j.err

# phylogenyAI Phase 2 — genoaligner QC pass over genes_final.
#
# The pipeline's curation (curate_gene_fastas.py) is length-only. This job
# adds the homology/coding QC genoaligner is built for: every record is
# aligned pairwise to its gene's bait, and each CIGAR is walked for
#   * homology screen   (score / coverage -> NO_ALIGN, LOW_COV)
#   * frameshifts       (indel runs % 3 != 0 -> FS=n)
#   * in-frame stops    (ref-codon-guided translation -> STOP=n)
# Genetic code 2 (vertebrate mito) for the 4 mtDNA genes, code 1 else.
# The tool only REPORTS; dropping records stays a pipeline decision.
#
# Output: data/gene_qc/<GENE>.qc.tsv + summary.tsv in the phylogenyAI tree.
set -o pipefail

PAI=/beegfs/a474r867/phylogenyAI
REPO=/beegfs/a474r867/genoaligner/repo
OUT=${OUT_DIR:-$PAI/data/gene_qc}
# BAIT_DIR: which bait set to use. baits_qc2 > baits_qc > baits.
BAIT_DIR=${BAIT_DIR:-}
# GENES: space-separated subset to run; empty = all of genes_final.
GENES=${GENES:-}
mkdir -p "$OUT"
cd "$REPO" || exit 1

echo "=== host: $(hostname)  $(date) ==="
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -pthread -I. -Iinclude \
    -o /tmp/gene_qc tools/gene_qc.cpp src/api/api.cpp || { echo "BUILD FAILED"; exit 1; }

# mtDNA genes get vertebrate-mitochondrial code (2); everything else code 1.
MTDNA=" COI CYTB ND1 ND2 "

echo -e "gene\tseqs\tOK\tNO_ALIGN\tLOW_COV\tFS\tSTOP\tinvalid" > "$OUT/summary.tsv"
RC=0
LIST="$GENES"
[ -z "$LIST" ] && LIST=$(ls "$PAI"/data/genes_final/*.fasta | xargs -n1 basename | sed 's/.fasta//')
for GENE in $LIST; do
    FA="$PAI/data/genes_final/${GENE}.fasta"
    [ -s "$FA" ] || { echo "$GENE: no genes_final fasta, skipped"; continue; }
    BAIT=""
    if [ -n "$BAIT_DIR" ]; then
        BAIT="$PAI/data/${BAIT_DIR}/${GENE}.baits.fasta"
    else
        for cand in baits_qc2 baits_qc baits; do
            [ -s "$PAI/data/$cand/${GENE}.baits.fasta" ] && { BAIT="$PAI/data/$cand/${GENE}.baits.fasta"; break; }
        done
    fi
    [ -s "$BAIT" ] || { echo "$GENE: no bait, skipped"; continue; }
    CODE=1; [[ "$MTDNA" == *" $GENE "* ]] && CODE=2
    echo; echo "### $GENE (code $CODE) ###"
    if ! /tmp/gene_qc "$FA" --ref "$BAIT" --code "$CODE" \
            --emit "$OUT/${GENE}.qc.tsv" 2>"$OUT/${GENE}.log"; then
        echo "$GENE: FAILED"; cat "$OUT/${GENE}.log"; RC=1; continue
    fi
    cat "$OUT/${GENE}.log"
    # summary.tsv row: gene + counts parsed from the tool's summary line
    S=$(grep -o 'OK=[0-9]* NO_ALIGN=[0-9]* LOW_COV=[0-9]* FS=[0-9]* STOP=[0-9]*' "$OUT/${GENE}.log")
    N=$(grep -c '^>' "$FA")
    INV=$(grep -o 'invalid=[0-9]*' "$OUT/${GENE}.log" | cut -d= -f2)
    echo -e "$GENE\t$N\t$(echo "$S" | tr ' ' '\t')\t$INV" >> "$OUT/summary.tsv"
done

echo; echo "=== summary ==="
column -t "$OUT/summary.tsv" 2>/dev/null || cat "$OUT/summary.tsv"
[ "$RC" -eq 0 ] && echo "PAI-QC: PASS" || echo "PAI-QC: FAIL"
exit "$RC"
