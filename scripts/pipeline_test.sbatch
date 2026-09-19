#!/bin/bash
#SBATCH --job-name=pipe_test
#SBATCH --partition=sixhour
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/beegfs/a474r867/genoaligner/tmp/pipe_test_%A_%a.out
#SBATCH --error=/beegfs/a474r867/genoaligner/tmp/pipe_test_%A_%a.err
#
# Pre-backbone pipeline test harness. One array task per gene; each task
# runs the configured stages via pipeline_test.py.
#
# Env knobs:
#   WORK           run dir (metrics.tsv lands here)          [required]
#   GENELIST       space-separated list, e.g. "CYTB ND1"     [required]
#                  (NOT comma-separated: sbatch --export splits on commas)
#   STAGELIST      space-separated stages (like GENELIST -- comma lists
#                  get split by sbatch --export and silently truncate)
#                  subset of: prefilter ortho extract align trim [align trim]
#   ALIGN_VARIANT  macse|genomsa|genomsa_lf|mafft            [genomsa]
#   TRIM_MODE      clipkit mode                              [smart-gap]
#   STRIP_FS       1 => replace '!' with '-' in trimmed out  [off]
#   GENES_SRC      dir with {GENE}.fasta or {GENE}/sample.fasta
#   RECORDS_SRC    dir with {GENE}.records.fasta (extract)
#   BAITS_SRC      dir with {GENE}.baits.fasta (ortho/extract)
#   BLASTDB        blast db prefix (ortho)
#   GENOMSA_ARGS   extra genomsa flags (e.g. "--refine 2")
#
# Example (align+trim on the 8-gene pilot):
#   sbatch --array=0-7 --export=WORK=$P/results/pipe_test/lf,GENELIST="APOB BDNF BRCA1 COI CYTB GHR IRBP RAG1A",ALIGN_VARIANT=genomsa_lf,GENES_SRC=$P/data/pilot_200_8genes pipeline_test.sbatch

set -o pipefail
REPO=/beegfs/a474r867/genoaligner/repo
ENV=/beegfs/a474r867/miniconda3/envs/genoml-phylo
export PATH="$ENV/bin:$PATH"

read -ra GL <<< "${GENELIST:?set GENELIST}"
GENE=${GL[$SLURM_ARRAY_TASK_ID]}
STAGES_ARG="${STAGELIST:-align trim}"
STAGES_ARG="${STAGES_ARG// /,}"

# per-task work subdir shares the metrics file via append
"$ENV/bin/python3" "$REPO/scripts/pipeline_test.py" \
    --work "${WORK:?set WORK}" \
    --genes "$GENE" \
    --stages "$STAGES_ARG" \
    --align-variant "${ALIGN_VARIANT:-genomsa}" \
    --trim-mode "${TRIM_MODE:-smart-gap}" \
    ${STRIP_FS:+--strip-fs} \
    --genes-src "${GENES_SRC:-}" \
    --records-src "${RECORDS_SRC:-}" \
    --baits-src "${BAITS_SRC:-}" \
    --blastdb "${BLASTDB:-}" \
    --genomsa "${GENOMSA_BIN:-/beegfs/a474r867/phylogenyAI/data/sim_precision/bin/genomsa_cpu}" \
    --genomsa-args "${GENOMSA_ARGS:-}" \
    --macse "${MACSE:-macse}" \
    --macse-args "${MACSE_ARGS:--max_refine_iter 0}" \
    --macse-lr-len "${MACSE_LR_LEN:-10000}" \
    --acc-csv "${ACC_CSV:-}" \
    --acc-fasta "${ACC_FASTA:-}" \
    --species-file "${SPECIES_FILE:-}" \
    --max-len "${MAX_LEN:-0}" \
    --median-mult "${MEDIAN_MULT:-0}" \
    --timeout "${TIMEOUT:-5400}" \
    --extract-script "${EXTRACT_SCRIPT:-/beegfs/a474r867/phylogenyAI/scripts/extract_gene_regions.py}"

echo "PIPE_TEST_DONE $GENE"
