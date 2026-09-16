# CASTER backbone experiment

This experiment evaluates CASTER-site as a fast, coalescence-aware topology
estimator for the Upham mammal matrices. It runs independently on the genomsa,
genomsa-lf, and codon supermatrices, plus the 15 completed mini-backbone-v2
cells.

Upham is used only after inference as a comparison reference. The CASTER
commands do not pass `--constraint` or `--guide`.

## Reproducible build

The build script checks out ASTER tag v1.25 at commit
`db2b3e95da5bb0318b933afe1a144eb943ef7cbf` and compiles `caster-site` with
the cluster C++ compiler:

```bash
cd /beegfs/a474r867/genoaligner/repo
bash scripts/caster_build.sh
```

The binary is installed under
`/beegfs/a474r867/phylogenyAI/tools/ASTER-v1.25/bin/caster-site`.

## Submit

```bash
cd /beegfs/a474r867/genoaligner/repo
full_job=$(sbatch --parsable scripts/caster_full_array.sbatch)
mini_job=$(sbatch --parsable scripts/caster_mini_array.sbatch)
report_job=$(sbatch --parsable \
  --dependency=afterok:${full_job}:${mini_job} \
  scripts/caster_report.sbatch)
printf 'full=%s mini=%s report=%s\n' "$full_job" "$mini_job" "$report_job"
```

The full matrices use the `kbs` partition because CASTER has no checkpoint
restart. The minis use a 15-task array on `sixhour`. Every run uses seed 233,
records wall time and maximum resident memory, and keeps completed outputs on
resubmission.

## Outputs

- Full trees: `results/caster_backbone/full/<variant>/caster.treefile`
- Mini trees: `results/caster_backbone/mini_v2/rep<N>/<variant>/caster.treefile`
- Runtime table: `results/caster_backbone/report/caster_runs.tsv`
- RF table: `results/caster_backbone/report/caster_rf.tsv`
- RF summary: `results/caster_backbone/report/caster_rf_summary.tsv`

RF comparisons are restricted to shared taxa and always report the shared
taxon count. The lf result is compared with every available IQ-TREE seed tree.

Any biological conclusion remains preliminary until the three full CASTER
runs finish and their topology sensitivity across alignment variants is
reviewed. This experiment does not establish equivalence between genomsa and
MACSE.
