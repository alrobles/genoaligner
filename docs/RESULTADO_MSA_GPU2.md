# RESULTADO — MSA-GPU Hito 2: kernel wavefront + pipeline completo 31 genes

Fecha: 2026-09-13. Continuación de RESULTADO_MSA_GPU1.md.

## Kernel wavefront (bloque-por-par, anti-diagonal)

`msa_pp_trace_kernel_wf` — un bloque de 256 hilos por par; los hilos barren
las anti-diagonales en paralelo (celda (i,j) en d=i+j solo lee d−1 y d−2 →
una barrera de bloque entre diagonales es toda la sincronización).
Recurrencia, direction-bytes, orden del endpoint-scan y walk **idénticos**
al kernel serial — solo cambia el orden de recorrido, que no puede cambiar
ningún valor. Buffers rodantes: 3 diagonales × 3 estados × (M+1) floats +
snapshots de borde, en scratch global empaquetado.

## Paridad (la regla del proyecto, cumplida)

| gate | resultado |
|---|---|
| Paridad par-a-par shim: wf vs ref (perfiles mergeados, IUPAC, fragmentos, global; ~300 casos, `run_block` con 256 hilos reales y barreras reales) | PASS bit-exacto |
| Driver parity shim (driver real con wf por defecto) | PASS bit-exacto |
| Dispositivo MI210: wf == serial (`KERNELS-AGREE`) | PASS byte-idéntico |
| Dispositivo MI210: GPU == CPU-ref (`PARITY`) | PASS byte-idéntico |

## Rendimiento en dispositivo (MI210, gfx90a)

| gen | n | align serial | align wavefront | speedup |
|---|---|---|---|---|
| BMI1 | 140 | 2.61s | 0.35s | 7.5× |
| COI | 1608 | 113.7s | 1.93s | **59×** |
| CYTB | 3523 | 102.4s | 3.37s | **30×** |

Los 31 genes de `genes_qc_pass` (19,156 secuencias) alineados en una sola
corrida: **Σ ≈ 205 s** (job 29231974). Tiempos por gen en
`results/msa_gpu/timings.tsv`. Los más lentos: CYTB 99.7s (host: dist 35.8s
+ NJ 60.4s, kernel solo 3.4s — corridas previas a `_mt`), BRCA1 14.8s,
VWF 12.4s, COI 11.5s.

## Etapas host paralelas (bit-exactas)

`kmer_distances_mt` y `nj_tree_mt` — paralelismo sin reordenar ninguna suma
flotante: distancias escriben en slots disjuntos; NJ reparte filas (cada
r[a] acumula en orden secuencial) y el argmin de Q fusiona mínimos
parciales en orden de escaneo (mismo desempate que serial). Test añadido
al gate: `_mt` == secuencial bit-exacto para n=2..40. El driver las usa
con `GENOMSA_THREADS` (default = hw).

## Pipeline downstream (mismo que MACSE, paso a paso)

Job 29231977 (28 s): `clipkit -m smart-gap` 31/31 → `pai_supermatrix.py`
→ **supermatrix 4353 taxa × 96,718 sitios, 31 particiones**
(`data/supermatrix_genomsa/`). IQ-TREE `-p -m MFP+MERGE -B1000` encadenado
(job 29231978, kbs) — espejo del pipeline MACSE.

## Comparación con MACSE (16 genes con salida MACSE disponible)

`results/msa_gpu/compare.tsv`: SPS (muestreada a 200 taxa) y SP-cost bajo
objetivo neutro (mismatch=1, gap=1, gap-gap=0).

- SPS alto en CDS conservados: VWF 0.977, ND2 0.811, BDNF 0.730.
- SPS bajo en loci no-CDS (intrones, genuinamente ambiguos): APP 0.165,
  CREM 0.178, FBN1 0.163, PLCB4 0.203 — confirma el hallazgo del QC.
- SP-cost: repartido (~mitad para cada método); ND2 favorece claramente a
  MACSE (codon-aware en CDS limpio): 7.53M vs 10.6M.

## Lectura honesta

- El kernel DP ya no es el cuello: en CYTB el alineamiento cuesta 3.4s de
  99.7s; dist+NJ en host dominan (ya mitigado con `_mt`, pendiente medir).
- genomsa es un alineador progresivo clásico (Clustal-like): kmer-NJ +
  Gotoh perfil-perfil. MACSE es codon-aware; en CDS limpio conserva fase
  mejor (ND2). La comparación definitiva es el árbol: IQ-TREE genomsa vs
  MACSE corriendo en paralelo.
- Memoria: dir_peak CYTB = 1.3 GB; VWF width 16.6k columnas — la matriz de
  direcciones crece con el ancho al cuadrado; para perfiles muy anchos
  hará falta traceback con checkpoints (Hirschberg) o dirs en diagonal.

## Pendiente

- RF/topología: árbol genomsa vs árbol MACSE vs árbol Upham.
- GPU-resident merges + subir kmer/NJ a GPU si el tamaño lo exige.
- Codon-aware scoring (opción B del survey: penalizar stops/frameshifts
  en col_score para loci CDS — mejora barata que cierra la brecha ND2).
