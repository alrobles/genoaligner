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

## Benchmark contra verdad conocida (scripts/msa_sim_truth.py)

La paridad bit-exacta certifica que GPU == referencia CPU, pero no que el
alineamiento sea biológicamente correcto. El benchmark de verdad simula
un ancestro de 1500 bp evolucionando por un árbol balanceado con
sustituciones e indels; cada base lleva su columna verdadera en una lista
global ordenada de columnas (las inserciones crean columnas nuevas
inmediatamente tras su ancla — el orden global es consistente con el
orden interno de cada hoja). SPS = fracción de pares homólogos
verdaderos reproducidos en el candidato.

Nota: dos versiones previas del simulador renderizaban inserciones fuera
de orden y reportaban SPS ~0.05 en alineamientos ~97% correctos — bug de
la verdad, no del alineador. La versión actual se auto-valida
(self-SPS = 1.0; ungapped(fila) == secuencia hoja, asertado).

Resultados (semilla 4, 2% subs + 0.4% indels por rama → ~12% divergencia
terminal):

| n   | SIM-SPS | width_true | width_got |
|-----|---------|------------|-----------|
| 2   | 0.9953  | 1521       | 1521      |
| 4   | 0.9928  | 1556       | 1552      |
| 8   | 0.9790  | 1620       | 1607      |
| 16  | 0.9753  | 1809       | 1765      |
| 32  | 0.9360  | 2060       | 1932      |
| 64  | 0.8964  | 2563       | 2208      |

Robustez: n=16 con semillas 7/9 → 0.980 / 0.968. Régimen duro
(5% subs + 1% indels por rama, ~30% divergencia): n=16 → 0.841.

Lectura: el pairwise DP es casi exacto; la pérdida es progresiva con n —
los merges del árbol guía acumulan errores de registro en regiones
repetitivas/ambigüedad de gaps, el comportamiento esperado de un MSA
progresivo. No es un defecto de implementación: es el techo del método
Clustal-like. La brecha a n grande sugiere iterar sobre refinement
(consistency / iterative realignment) si se quiere subir el techo.

## Mejora de calidad: PSGP + gappy-strip (post-TWILIGHT audit)

Tras la comparación con TWILIGHT (`RESULTADO_MSA_TWILIGHT.md`), se
implementaron sus dos heurísticos de scoring en genomsa (commit
`969c274`), ambos ON por defecto:

- **PSGP** (`Params::psgp`, convención ClustalW/TWILIGHT): una columna que
  ya contiene gaps acepta gaps nuevos más barato —
  `open[c] = occ==1 ? go : max(0.1·go, 0.5·go·occ)`,
  `ext[c] = occ==1 ? ge : max(0.2·ge, ge·occ)`. Implementado en la
  referencia (`msa.hpp` = spec) e idénticamente en ambos kernels
  (`pp_open`/`pp_ext` desde el array `occ`, sin cambio de layout).
- **Gappy-strip** (`Params::gappy = 0.95`): runs contiguos de columnas con
  fracción de gap > umbral se quitan antes del DP y se reinsertan en el
  CIGAR como bloques de inserción; runs coincidentes en ambos perfiles se
  mini-alinean globalmente. El driver GPU strippea antes de empaquetar y
  expande tras el download → el kernel ve perfiles reducidos, paridad
  estructural.

Sim-truth n=64, mismo benchmark que arriba:

| cfg | SIM-SPS | width (true) |
|-----|---------|--------------|
| base (ambos off)      | 0.8831 | 2300/2800 |
| psgp solo             | 0.9299 | 2513/2800 |
| gappy 0.95 solo       | 0.9045 | 2392/2800 |
| **psgp + gappy**      | **0.9436** | 2603/2800 |

Semillas 7/9: 0.9018→0.9443, 0.8810→0.9470. Determinista (runs repetidos
byte-idénticos). El alineamiento ya no sobre-compacta (2300→2603 vs
verdad 2800). vs TWILIGHT 0.963: cierra ~65% de la brecha restante;
lo que queda es probablemente su modelo de inserciones + refinement.

Paridad: `msa_pp_parity`, `msa_pipeline_parity`, `msa_driver_parity` con
variantes default/legacy/psgp/gappy/psgp+gappy — todas bit-exactas.
Flags: `--psgp/--no-psgp`, `--gappy T/--no-gappy`.

## Lectura honesta

- El kernel DP ya no es el cuello: en CYTB el alineamiento cuesta 3.4s de
  99.7s; dist+NJ en host dominan (ya mitigado con `_mt`, pendiente medir).
- genomsa es un alineador progresivo clásico (Clustal-like): kmer-NJ +
  Gotoh perfil-perfil. MACSE es codon-aware; en CDS limpio conserva fase
  mejor (ND2). La comparación definitiva es el árbol: IQ-TREE genomsa vs
  MACSE corriendo en paralelo.
- Contra verdad simulada recupera ~97-99% de pares homólogos en régimen
  de ortólogos de mamífero (n≤16) y ~90% a n=64 — evidencia de calidad
  real, no solo de paridad interna.
- Memoria: dir_peak CYTB = 1.3 GB; VWF width 16.6k columnas — la matriz de
  direcciones crece con el ancho al cuadrado; para perfiles muy anchos
  hará falta traceback con checkpoints (Hirschberg) o dirs en diagonal.

## Pendiente

- RF/topología: árbol genomsa vs árbol MACSE vs árbol Upham.
- GPU-resident merges + subir kmer/NJ a GPU si el tamaño lo exige.
- Codon-aware scoring (opción B del survey: penalizar stops/frameshifts
  en col_score para loci CDS — mejora barata que cierra la brecha ND2).
