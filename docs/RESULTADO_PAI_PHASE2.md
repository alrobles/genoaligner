# RESULTADO — phylogenyAI Fase 2: QC por homología + filtrado + MSA

**Fecha**: 2025-XX (jobs 29231071, 29231076, 29231081, 29231093, 29231335,
29231439, 29231447)
**Datos**: `phylogenyAI/data/genes_final/` — 31 genes, 19,366 records (MamPhy
accessions, conjunto Upham et al. 2019)
**Herramienta QC**: `tools/gene_qc` (genoaligner SW + CIGAR, MI210)

## Diseño

genoaligner no hace MSA. Su papel en Fase 2 es el screen pairwise
record-vs-bait que la curación previa (solo longitud) no hacía:

- **homología** (score, cobertura del span alineado)
- **frameshifts** (`fs_net` = net indel mod 3 sobre el CIGAR)
- **stops internos in-frame** (codones referencia-guiados, contiguos)
- código genético por locus: 1 nuclear, 2 mitocondrial (COI/CYTB/ND1/ND2)

## Resultados del QC

| pasada | baits | records | tiempo (MI210) |
|---|---|---|---|
| QC original | `baits/` | 19,366 | ~4:40 min (31 genes) |
| QC same-locus | `baits_qc2/` (13 genes) | subset | — |
| **QC final** | **`baits_final/`** | 19,336 | ~5 min |

### Hallazgo estructural: loci con intrones

De los 13 genes con bait "sucio" (stops en todo frame), **9 no tienen
ningún record en `genes_final` que cubra ≥70% del locus y traduzca
limpio**: APP, BMI1, CREM, FBN1, GHR, ND1, PLCB4, TYR1, PNOC. La región
ampliada no es CDS puro (intrones/UTR) → el check de stops/FS **no aplica
por construcción** en esos loci (sus FS ~50% son indels de intrón, no
defectos). Para esos genes el QC queda homology-only; la limpieza
defiere al paso de gene-trees (como en Upham).

Para BCHE, ENAM, RAG2, COI sí existió record same-locus limpio → esos
baits se reemplazaron en `baits_final/`.

### Corroboración phase-free (clave de honestidad)

Los flags FS/STOP del alineamiento **no son prueba suficiente**: un indel
no-triplete colocado al borde del span (o en flanco UTR alineado) pone
`fs_net≠0` sin frameshift real, y una ventana de fase desplazada genera
codones-stop espurios. Verificado en datos reales:

- **ND2**: records flagged FS traducen limpio en su propio frame
  (1 stop terminal) → artefacto de colocación, no defecto.
- **CYTB**: de 587 records STOP-flagged, solo ~10 tienen stops internos
  propios → la gran mayoría eran artefactos de fase del alineamiento.

Por eso `scripts/pai_qc_filter.py` exige corroboración: solo se dropea
un record flagged si **su propia traducción** (mejor de 3 frames,
excluyendo el codón terminal) tiene stops internos.

## genes_qc_pass/

| métrica | valor |
|---|---|
| records totales | 19,336 |
| **kept** | **19,156 (99.1%)** |
| dropped NO_ALIGN | 1 (BRCA1) |
| dropped corroborated FS/STOP | 179 |
| por gen (ej.) | CYTB 3559→3523, IRBP 1284→1252, ENAM 171→153 |

Drop lists por gen + razones en `genes_qc_pass/*.drop.tsv`;
`summary.tsv` agregado. LOW_COV se conserva (fragmentos legítimos).

## MSA: MACSE + ClipKIT sobre el set filtrado

`align_genes_qc_array.sbatch` (31 tareas): MACSE v2 codon-aware +
`exportAlignment` (stops/FS → NNN) + ClipKIT smart-gap →
`data/alignments_qc*/`, stats por seq en `*.stat_per_seq.csv`
(auditoría de segundo nivel: internal_FS/internal_STOP/internal_DEL).

**Corrección de correctitud**: MACSE corría con código estándar para
todos los genes; los 4 mitocondriales necesitan `-gc_def 2` (vertebrate
mitochondrial — TGA=Trp). Las 4 tareas se re-enviaron con el código
correcto (`align_mt_qc_array.sbatch`, job 29231335).

**Timeout**: 7 genes (APOB, BRCA1, BRCA2, GHR, IRBP, RAG1A, RAG1B)
excedieron el límite de `sixhour` (5:50) en MACSE → re-enviados a `kbs`
72h/8 cores (`align_retry_qc_array.sbatch`, job 29231439).

## Pendiente al momento de escribir

- Completar los 31 trimmed alignments (arrays 29231335 + 29231439)
- Supermatrix + particiones (job 29231447, `scripts/pai_supermatrix.py`)
- Árbol ML global IQ-TREE2 particionado (script preparado)

## Lecciones

1. Un flag de alineamiento no es un diagnóstico: exigir corroboración
   independiente (traducción phase-free) antes de descartar datos.
2. El bait correcto no es "el record más limpio" sino "el record limpio
   que cubre el mismo locus" — la primera selección por limpieza eligió
   isoformas/regiones distintas (APP: LOW_COV=356/359 artificial).
3. El código genético es un parámetro de correctitud, no un detalle:
   MACSE con código estándar sobre mtDNA introduce frameshifts falsos.
4. MACSE no escala trivialmente: >5:50h para genes de ~500-1300 records
   largos; planificar partición larga o MAFFT+traducción para esos.
