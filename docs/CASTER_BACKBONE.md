# CASTER backbone experiment

This experiment evaluates CASTER-site as a fast, coalescence-aware topology
estimator for the Upham mammal matrices. It runs independently on the genomsa,
genomsa-lf, and codon supermatrices, plus the 15 completed mini-backbone-v2
cells.

Upham is used only after inference as a comparison reference. The CASTER
commands do not pass `--constraint` or `--guide`.

## Reproducible build

The build script checks out ASTER tag v1.25 at commit
`db2b3e95da5bb0318b933afe1a144eb943ef7cbf`. Its default `portable` profile
does not use `-march=native`, so one binary can run across heterogeneous CPU
nodes:

```bash
cd /beegfs/a474r867/genoaligner/repo
bash scripts/caster_build.sh
```

The binary is installed as
`/beegfs/a474r867/phylogenyAI/tools/ASTER-v1.25/bin/caster-site-portable`.
Set `CASTER_BUILD_PROFILE=strict` for deterministic floating-point validation,
or `CASTER_BUILD_PROFILE=native` only when the build and compute CPUs are
guaranteed to expose the same instruction set. Each binary has an adjacent
`build.tsv` provenance file. Select a non-default profile at runtime with
`CASTER_BIN=/path/to/caster-site-strict`.

## Submit

```bash
cd /beegfs/a474r867/genoaligner/repo
bash scripts/caster_submit.sh
```

The submitter refuses to start while another CASTER full or mini array is
active, selects only missing cells, and runs the report with `afterany`.
Partial failures therefore remain visible instead of leaving the report
permanently blocked. The full matrices use the `kbs` partition because CASTER
has no checkpoint restart. The minis use `sixhour`. Every run uses seed 233,
records binary identity, backend, host, wall time and maximum resident memory,
and keeps completed outputs on resubmission.

For CPU allocation tuning on all 4,353 taxa, create a deterministic 30,000-site
benchmark and run the scaling array:

```bash
python3 scripts/caster_sample_alignment.py \
  --input /beegfs/a474r867/phylogenyAI/data/supermatrix_genomsa/supermatrix.fasta \
  --output /beegfs/a474r867/phylogenyAI/results/caster_backbone/benchmark/supermatrix-30000.fasta \
  --sites 30000
sbatch scripts/caster_cpu_scale.sbatch
```

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
