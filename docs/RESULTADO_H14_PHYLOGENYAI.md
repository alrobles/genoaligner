# H14 — genoaligner vs MAFFT sobre datos reales de phylogenyAI (CYTB)

Jobs: **29230759** (completo) + **29230973** (verificación, REUSE_DIR).
Cluster KU HPC, nodo `r06r18n01`, GPU MI210 (ROCm 6.4.3), MAFFT 7.526
(`genoml-phylo` conda env — el mismo que usa el pipeline de Upham).

## La tarea, declarada de antemano

**"Alinear cada secuencia de un gen contra una referencia, pairwise"** —
la forma exacta del paso de ortología/RBH donde un alineador pairwise
compite. genoaligner NO es un alineador múltiple; esto no es un reemplazo
de MAFFT para MSA. Es la comparación honesta: la misma tarea pairwise,
los mismos datos reales, medidos en el mismo job en el mismo nodo.

- Entrada: `phylogenyAI/data/genes_final/CYTB.fasta` — 3559 secuencias
  reales, 207–1590 bp.
- Referencia: registro 0 (`AB002412.1`, 1137 bp), declarado.
- Tarea: **3558 alineamientos pairwise** vs la referencia (4315.6 Mcells).
- Esquema de score: `{match=1, mismatch=-1, gap_open=2, gap_extend=1}` —
  el esquema fijado por el oráculo SeqAn3.
- MAFFT invocado exactamente como el pipeline lo hace:
  `mafft --auto --thread N` sobre FASTAs de 2 secuencias
  (`scripts/align_genes_array.sbatch`).

## Rendimiento — la misma tarea, el mismo nodo

| método | wall time para 3558 pares | pares/s | GCUPS |
|---|---|---|---|
| **genoaligner + CIGAR** (MI210) | **2.06 s** (mean, n=5) | 1729 | 2.10 |
| **genoaligner score-only** (MI210) | **1.06 s** (mean, n=7) | 3355 | 4.07 |
| mafft --auto serial (1 par/proceso) | 2258 s | 1.6 | — |
| mafft --auto, 8 procesos | 323 s | 11.0 | — |

Speedup vs MAFFT: **~1100×** (serial) / **~157×** (8-way) con CIGAR
incluido; **~305×** vs 8-way score-only.

Notas de honestidad metodológica:
- El tiempo de MAFFT incluye spawn de proceso + I/O por par — ese ES el
  costo real de usarlo para esta tarea (no tiene modo batch-pairwise),
  pero se declara.
- genoaligner se midió con un co-tenant en otra GPU del nodo: spread
  18–59% entre reps. Se reporta el mean, no el min.
- Un gen (CYTB, el más grande del pipeline). Otros genes darían números
  distintos; la dirección del resultado no depende del gen.

## Correctitud sobre datos reales

**SeqAn3 (oráculo externo, mismos 3558 pares):**

| subconjunto | filas | acuerdo de score |
|---|---|---|
| pares ACGT-puro | 3242 | **3242/3242 — 100% exacto** |
| pares con códigos IUPAC ambiguos | 316 | 0 — ver explicación |
| total | 3558 | 3242 (91.1%) |

**Las 316 discrepancias están completamente explicadas** — no son errores:

- CYTB.fasta contiene códigos de ambigüedad IUPAC (6356 N, 250 Y, 118 R,
  81 M, 45 K…). TODAS las filas discrepantes contienen al menos uno;
  NINGÚN par ACGT-puro discrepa.
- `nucleotide_scoring_scheme` de SeqAn3 otorga crédito parcial cuando los
  conjuntos IUPAC se solapan (R∩A, etc.). genoaligner compara literales:
  una base ambigua solo empata con el mismo código.
- Dirección del efecto: **genoaligner nunca puntúa más alto que SeqAn3**
  (0 pares `ours > seqan3`). Es estrictamente más conservador.
- 188 pares CON ambigüedades sí concuerdan — las posiciones ambiguas no
  tocan el alineamiento óptimo.
- Semántica declarada: genoaligner hace comparación literal de bases.
  Soporte de matching IUPAC parcial queda como decisión de diseño futura.

**CIGARs**: los 3558 resultados re-verificados dentro de la API
(`rescore_ok` + `wellformed_ok` por par): 0 inválidos, 0 `too_large`,
0 pares sin alinear (toda secuencia real tiene bloque conservado vs la
ref de CYTB).

**Concordancia biológica con MAFFT:**

- Mediana de |Δidentidad| vs `mafft --auto` (global): **0.015** —
  acuerdo fuerte donde los objetivos coinciden (homólogos de longitud
  similar).
- 529 pares (14.9%) con Δ>0.10: **secuencias parciales** (ej. fragmento
  de 228 bp vs ref de 1137 bp). `mafft --auto` produce alineamiento
  GLOBAL: rellena el fragmento con ~900 columnas de gaps → su "identidad
  global" (~19%) subestima la similitud. genoaligner (local) coloca el
  fragmento: 97.4% de identidad, cobertura 100%.
- **Verificación de la colocación**: `mafft --localpair` sobre los peores
  pares reproduce los spans de genoaligner **exactamente**:
  par 1119 → ambos ref 91..318, id 0.974; par 1436 → 98..349;
  par 1676 → 0..206; par 3230 → 256..468.

## Incidentes del benchmark (registrados para el récord)

1. **3558/3558 discrepancias SeqAn3 en la primera corrida** — el driver
   emitió scores `{2,-3,3,1}` y el oráculo fija `{1,-1,2,1}` (el TSV no
   lleva columna de esquema). Discrepancia de escala que el oráculo
   detectó correctamente — evidencia de que el gate externo funciona.
   Corregido: esquema por defecto = el del oráculo, `--scoring` lo cambia.
2. **`geno_id=0.000` en todos los pares** — la comparación parseaba CIGAR
   run-length (`28M`); los nuestros son ops expandidos (`MMM…`).
   Corregido.
3. **Bug de symlink en REUSE_DIR** — `ln -sfn` sobre un dir existente
   anida el enlace. Corregido con `rm -rf` previo.

## Lo que el milestone del paper puede afirmar — y lo que no

Sí: *"En la tarea pairwise-vs-referencia (la forma de ortología/RBH) sobre
los datos reales del pipeline, genoaligner produce los 3558 alineamientos
con CIGAR en ~2 s en una MI210 — ~150–1100× el wall time de MAFFT por par
— con scores verificados exactos por SeqAn3 en todo par sin códigos de
ambigüedad, y colocaciones de fragmentos idénticas a `mafft --localpair`."*

No: *"genoaligner reemplaza a MAFFT"* (es pairwise, no MSA);
*"genoaligner siempre coincide con MAFFT"* (objetivos local vs global
difieren legítimamente; IUPAC difiere por semántica declarada).

## Reproducir

```bash
GENE=CYTB sbatch scripts/h14_vs_mafft.sbatch            # corrida completa
REUSE_DIR=/path/to/prev_run sbatch scripts/h14_vs_mafft.sbatch  # reusar mafft
```

Artefactos: `vs_mafft_<job>/geno_cigar.tsv` (resultados + CIGAR),
`oracle_cigar.tsv` (columnas SeqAn3), `pairs/*.aln` (salidas mafft).
