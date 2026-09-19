# RESULTADO H13-FLAT-MERGE — el flat kernel pasa a ser el score kernel por defecto

Fecha: 2026-09-12 · Rama: main · Commit: e95316a
Job GPU: **29230707** (MI210, r06r26n01, `gres=gpu:mi210:1`)

## La decisión

`wfa_score_kernel_flat` (bloque fijo de 128 hilos, grid-stride interior) es ahora
el kernel que `align_batch` lanza en el path score-only — **sin flag**. La
opción considerada y descartada: exponerlo como opción de API (un usuario no
debería elegir kernels) o dejarlo detrás de `WFA_API_USE_FLAT` (un candidato
medido y verificado no es una decisión, es una dilación).

El kernel original **no se borra**: es el baseline que el gate H2
(`tests/parity/wfa_parity.cpp`) y el bench `tcus` siguen ejercitando — si el
flat regresara, la comparación seguiría existiendo.

## Por qué es seguro fusionarlo

- **Misma álgebra, mismo layout, mismos centinelas, mismo sobre de shared**
  (`2*(2smax+3)` ints): el flat es una reasignación de hilos, no una reescritura
  — `docs/RESULTADO_FASE7_OPTIMIZACION.md`.
- **Medido**: 1.49-2.67x en MI210, 1.58-4.41x en PRO 6000, ratios reproducibles
  al 0.5% — y elimina el modo inestable del original (oscilaba 4.4x dentro de
  un proceso por carriles ociosos en `__syncthreads`).
- **Verificado en GPU** desde Fase 7 (h7_flat.sbatch, `--verify` con hilos
  reales y `WFA_FLAT_BLOCK` barrido).
- **Contrato sin cambios**: `smax ≤ 511` se mantiene aunque el grid-stride del
  flat no lo necesita (el original sí lo necesitaba: un bloque < `2smax+1`
  saltaba diagonales en silencio). El contrato no se ensancha de callado.

## Qué se añadió para no depender de la evidencia vieja

- `bench/h7_flat_parity.cpp` ahora es **stage del gate CPU** (ambas pasadas):
  1225/1230 resueltos, 0 discrepancias vs la DP CPU independiente.
- `test_api` corre ahora un batch score-only sobre los mismos pares de la
  sección de correctitud y **verifica scores** contra la DP CPU — antes el
  path score-only solo se checaba a nivel shape en GPU, así que el merge
  habría salido verificado solo en shim.

## Evidencia del merge (job 29230707, MI210)

- `score-only path (flat kernel): mismatches=0` — el flat verificado
  end-to-end por la API en device.
- Real-seq score-only `smax=200/511`: verde (esos llamados ahora ejecutan flat).
- B4 batch semantics, concurrencia, SW completo, SeqAn3 150/150: todo verde.
- `H10: PASS` con el flat como kernel por defecto.
