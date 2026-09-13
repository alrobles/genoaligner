# phylogenyAI Fase 2 — QC por homología con genoaligner (31 genes)

Job: **29231028** (MI210, 4:40 min total incl. build). Herramienta:
`tools/gene_qc.cpp` — consumer de la API pública `align_sw_batch`.
Entrada: `phylogenyAI/data/genes_final/*.fasta` (19,366 records, ya
filtrados por longitud por `curate_gene_fastas.py`).
Referencia: `data/baits/<GENE>.baits.fasta` (mejor record+frame
autoseleccionados, ver abajo).
Salida: `phylogenyAI/data/gene_qc/<GENE>.qc.tsv` + `summary.tsv`.

## Qué mide (y lo que la curación actual NO medía)

La curación del pipeline es solo por longitud (min-length + cap por gen).
gene_qc añade, por record, usando el CIGAR del alineamiento local vs bait:

| señal | definición | flag |
|---|---|---|
| homología | score 0 → probable contaminante/mal etiquetado | `NO_ALIGN` |
| cobertura | span_text/len < 0.5 → fragmento | `LOW_COV` |
| frameshift duro | (ins−del) mod 3 ≠ 0 en el span → no cabe in-frame | `FS=` |
| frameshift suave | runs indel %3≠0 (ambiguo en repeats) | columna `fs_runs` |
| stop interno | codón stop in-frame, bases contiguas, no terminal | `STOP=` |
| stop terminal | stop en el último codón completo del span | col `stop_end` |

## Resultados (job 29231028)

| gene | n | OK | FS | STOP | nota |
|---|---|---|---|---|---|
| CYTB | 3559 | 2738 | 260 | 587 | bait limpio |
| COI | 1614 | 1398 | 175 | 55 | bait limpio |
| IRBP | 1284 | 1136 | 54 | 101 | bait limpio |
| ND2 | 960 | 356 | 417 | 366 | bait limpio |
| ND1 | 941 | 826 | 50 | 73 | bait sucio (33 stops) |
| GHR | 926 | 608 | 125 | 293 | bait sucio (17) |
| BRCA1 | 914 | 558 | 63 | 324 | bait limpio; 1 NO_ALIGN |
| RAG1B | 865 | 671 | 17 | 9 | bait limpio |
| RAG2 | 845 | 813 | 24 | 8 | bait limpio |
| VWF | 757 | 710 | 16 | 20 | bait limpio |
| RAG1A | 554 | 381 | 42 | 1 | bait limpio |
| APOB | 513 | 470 | 10 | 32 | bait limpio |
| BDNF | 460 | 451 | 7 | 1 | bait limpio |
| PLCB4 | 430 | 0 | 303 | 429 | **bait sucio (6)** |
| ATP7 | 404 | 374 | 28 | 22 | bait limpio |
| ADORA3 | 386 | 356 | 25 | 4 | bait limpio |
| PNOC | 365 | 322 | 29 | 21 | bait casi limpio (2) |
| APP | 359 | 0 | 224 | 358 | **bait sucio (12)** |
| DMP1 | 352 | 299 | 6 | 48 | bait limpio |
| CNR1 | 307 | 303 | 4 | 0 | bait limpio |
| TYR1 | 289 | 257 | 18 | 27 | bait sucio (17) |
| CREM | 283 | 1 | 189 | 282 | **bait sucio (9)** |
| TTN | 269 | 251 | 16 | 10 | bait limpio |
| EDG1 | 256 | 241 | 14 | 1 | bait limpio |
| FBN1 | 252 | 0 | 160 | 252 | **bait sucio (23)** |
| BCHE | 245 | 30 | 212 | 176 | bait casi limpio (1)* |
| A2AB | 238 | 204 | 27 | 8 | bait limpio |
| BRCA2 | 235 | 83 | 15 | 56 | bait limpio |
| ENAM | 171 | 0 | 138 | 160 | **bait sucio (14)** |
| ADRB2 | 163 | 162 | 0 | 0 | bait limpio |
| BMI1 | 140 | 0 | 119 | 139 | **bait sucio (4)** |

`bait_stops` = stops que el propio bait tiene en su mejor frame; >0 marca
la evidencia de stops como débil para ese gen (WARN en el log).

## Verificación de la señal (post-hoc, datos reales)

Para descartar que los flags fueran artefactos del método, se tradujeron
los records flagged en sus 3 frames propios y se separó el codón stop
**terminal** del CDS (biológicamente normal) de stops internos:

- **CYTB**: de 587 flagged → 467 con stop INTERNO genuino (candidatos
  NUMT/pseudogene/error — exactamente lo que la limpieza de Upham
  existe para quitar).
- **COI**: de 100 flagged → solo 8 internos; 92 eran el terminador
  normal. Motivo del refinamiento `stop_end`.
- **ND1**: 349→73 tras `stop_end` — la mayoría eran el terminador.
- **GHR/BRCA1**: 273/309 internos reales (incluye fragmentos genómicos
  con intrones: LOW_COV=92 en BRCA2).
- **FS** con bait limpio es señal real pero suave: `fs_net`≠0 es duro
  (p.ej. CYTB 260 records), `fs_runs` solo queda como columna de triage
  porque un indel in-frame en un repeat puede quedar partido en runs.

## Defectos del propio QC encontrados y corregidos en el proceso

1. **Frame fijo = 0** → STOP=100% espurio en genes cuyo bait no empieza
   en frontera de codón (RAG1A limpio solo en frame 1). Ahora el frame
   se autocalibra por bait-record (min stops, tie → ATG-start).
2. **Baits que no son CDS limpio**: 9 genes (APP, BMI1, CREM, ENAM,
   FBN1, PLCB4, GHR, ND1, TYR1) tienen `bait_stops` 4–33 → sus flags de
   stop son débiles por definición. **Acción para el pipeline**: esos
   genes necesitan mejores referencias (RefSeq CDS) antes de usar STOP.
3. **Workspace**: VWF/BRCA1 excedieron memoria device por chunk →
   reintento automático con batch a la mitad.
4. **Codón sobre inserción**: un codón solo cuenta si sus 3 bases de
   text son contiguas (un indel dentro lo invalida; ya es señal FS).
5. **Stop terminal**: el stop en el último codón del span va a
   `stop_end`, no al flag (92/100 de COI eran esto).

## Uso en el pipeline

- Decisión recomendada por gen sobre `genes_final`:
  - `NO_ALIGN`: revisar/remover (1 caso en BRCA1).
  - `STOP`/`FS` (net≠0): candidatos a remover en genes con bait limpio.
  - `LOW_COV`: no es defecto — son fragmentos parciales; conservarlos.
  - Genes con bait sucio: NO usar STOP/FS hasta mejorar la referencia;
    la tabla igual sirve para elegir un mejor bait (record con frame
    limpio y `bait_stops=0` si existe en el archivo de baits).
- La tabla `data/gene_qc/summary.tsv` es el deliverable auditable por gen.

## Reproducir

```bash
sbatch scripts/pai_gene_qc.sbatch   # en /beegfs/a474r867/genoaligner/repo
```

Limitaciones declaradas: matching literal de bases (códigos IUPAC no
dan crédito parcial — semántica conservadora, ver RESULTADO_H14);
fs_runs inflado por ambigüedad de colocación en repeats (fs_net es el
duro); `min_cov=0.5` y `min_score=0` por defecto son umbrales de
triage, no de rechazo.
