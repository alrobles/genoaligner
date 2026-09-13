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
OUT=$PAI/data/gene_qc
mkdir -p "$OUT"
cd "$REPO" || exit 1

echo "=== host: $(hostname)  $(date) ==="
/kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -pthread -I. -Iinclude \
    -o /tmp/gene_qc tools/gene_qc.cpp src/api/api.cpp || { echo "BUILD FAILED"; exit 1; }

# mtDNA genes get vertebrate-mitochondrial code (2); everything else code 1.
MTDNA=" COI CYTB ND1 ND2 "

echo -e "gene\tseqs\tOK\tNO_ALIGN\tLOW_COV\tFS\tSTOP\tinvalid" > "$OUT/summary.tsv"
RC=0
for FA in "$PAI"/data/genes_final/*.fasta; do
    GENE=$(basename "$FA" .fasta)
    # Prefer baits_qc (clean-CDS references chosen by stop-count) when present.
    BAIT="$PAI/data/baits_qc/${GENE}.baits.fasta"
    [ -s "$BAIT" ] || BAIT="$PAI/data/baits/${GENE}.baits.fasta"
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
