# RESULTADO — MSA-GPU Hito 1: driver completo + validación en dispositivo real

Fecha: 2026-09-13. Rama `main` @ `88fc693` + fix namespace.

## Lo construido (sprint M1–M3)

| pieza | archivo | estado |
|---|---|---|
| Referencia CPU (semántica canónica) | `src/msa/msa_ref.cpp` | PASS |
| Kernel perfil-vs-perfil (thread-por-par) | `include/genoaligner/backend/msa_pp_kernel*.hip` | PASS paridad |
| Scheduler por niveles del árbol guía | `tree_levels()` en `msa_ref.cpp` | PASS paridad |
| Driver GPU host (buffers empaquetados, 1 launch/nivel) | `src/msa/msa_gpu.cpp` | PASS paridad shim + dispositivo |
| CLI | `tools/genomsa.cpp` | operativo |

Semántica: kmer-Jaccard fragmentado → NJ determinístico → Gotoh semiglobal
perfil-vs-perfil (match 2, ts −1, tv −2, gap-open 3 escalado por ocupación,
ext 1, extremos libres) → merge determinístico → filas en orden de entrada.

## Paridad en dispositivo real (MI210, gfx90a)

Job 29231953, nodo r06r18n01:

```
### parity subset: BMI1 first 150 records ###
PARITY: PASS (GPU == CPU-ref, byte-identical, on-device)
### full BMI1 (GPU) ###   140 seqs, 20 levels, 18.4s
### full COI (GPU) ###    1608 seqs, 23 levels, 123.3s
                          (dist 4.1s, tree 5.4s, align 113.7s, dir_peak 598.6MB)
MSA-VAL: PASS
```

Checks estructurales: 0 filas ragged, 0 violaciones `ungap==input`
(las filas alineadas contienen exactamente las letras del input, nada
inventado ni perdido).

## Comparación con MACSE (BMI1, mismo set `genes_qc_pass`)

| | genomsa | MACSE | ClipKIT(MACSE) |
|---|---|---|---|
| width | 723 | 600 | 579 |
| gaps | 57.0% | — | — |
| SP-cost (mismatch=1, gap=1, gap-gap=0) | **503,242** | 577,923 | — |

- SPS (pares de residuos de genomsa reproducidos en MACSE): 0.433 —
  esperable: BMI1 es locus no-codificante (intrones), región genuinamente
  ambigua; métodos distintos difieren en colocación de gaps.
- **Bajo un objetivo SP neutro nuestro alineamiento es 13% mejor** —
  evidencia de que el DP perfil-perfil encuentra estructura conservada real,
  no un colapso de gaps.
- Tiempo: genomsa 18s vs MACSE ~30–60min por gen en `kbs` (y COI/CYTB
  excedieron 5:50h en sixhour). Incluso el kernel correctness-first es
  órdenes de magnitud más rápido en este tamaño.

## Bugs reales cazados por la disciplina de paridad

1. **Traceback flotante en la referencia**: re-derivar el predecesor como
   `v = mM[i][j] - col_score` y comparar contra los 3 estados falla por
   redondeo → caía al default `'D'` accidentalmente. Mismo score, CIGAR
   distinto = "plausible pero incorrecto". Fix: la referencia graba
   decisiones en direction-bytes durante el fill (misma especificación que
   el kernel).
2. **Orden del endpoint-scan**: ref intercalaba M/Ix por-i; kernel hacía 4
   fases separadas → empates resueltos distinto. Unificado: por-i (M,Ix)
   col N, por-j (M,Iy) row M, `>` estricto.
3. `ncols()` usado antes de poblar `cols` en merge → assert; corregido a
   largo de fila.
4. Filas en orden de árbol, no de entrada → `Profile::ids` + reorden final.

## Limitaciones honestas

- Kernel = un hilo por par: el DP es serial por par. GPU ~140× más lento
  que CPU-ref en BMI1 (18.4s vs 0.13s) — diseño correctness-first;
  M4 = wavefront/warp-per-par para el DP.
- NJ es O(n³) en CPU (5.4s @ n=1608; ~60s esperado @ 3523) — candidato a
  GPU si se vuelve cuello.
- Merge de perfiles en host (copias DtoH por nivel). Aceptable ahora;
  GPU-resident merges después.
- No codon-aware: la garantía biológica la da el QC upstream
  (stops/frameshifts filtrados), no el alineador. MACSE sigue corriendo
  como referencia independiente.

## Siguiente (M4)

- Kernel wavefront: bloque-per-par, anti-diagonal paralela, direction
  bytes igual (semántica inmutable), paridad obligatoria.
- Validación CYTB (3523 seqs) corriendo: job 29231959.
- Trimming de columnas (equivalente ClipKIT) — trivialmente paralelo.
