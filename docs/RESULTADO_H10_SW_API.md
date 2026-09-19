# RESULTADO H10-SW-API — SW en la API pública, Fase C

Fecha: 2026-09-12 · Rama: main · Commit: b25bf81
Job GPU: **29227600** (MI210, r07r16n01, nodo exclusivo, `gres=gpu:mi210:1`)

## Qué se construyó

SW entra a la API pública como **superficie separada**, no como un campo
`algorithm` sobre `AlignRequest`/`AlignResult`:

- `SWRequest` (text/pattern + `SWScoring{match,mismatch,gap_open,gap_extend}` +
  `with_cigar`) — **sin `smax`**: el alineamiento local no tiene presupuesto de
  distancia; el campo no se ignora, no existe.
- `SWAlignResult` con `score` (puntuación, mayor=mejor — `AlignResult.score` es
  distancia de edición, menor=mejor; sobrecargar el mismo campo habría sido el
  cambio-semántico-silencioso que el proyecto existe para evitar), coordenadas
  locales `start_i/start_j/end_i/end_j`, `cigar`, `rescore_ok`,
  `wellformed_ok`, `too_large`.
- `SWBatchResult` con `results` ordenados, conteos, `Status` y `error`.
- Entry points `align_sw()` y `align_sw_batch()`.
- `examples/align_sw.cpp` (target `genoaligner_example_sw`) — el path completo
  de usuario por la API pública, compilado y ejecutado por el gate.

## Decisiones de contrato (todas rechazan pre-device, todas en api.hpp)

- **`gap_extend > gap_open` + `with_cigar` → `invalid_argument`.** Es el
  hallazgo de Fase B: la DP re-abre gaps de 1 adyacentes y el CIGAR no puede
  re-puntuar a su propio score. Score-only lo acepta — el score sigue exacto.
- **Un esquema de scoring por batch** (análogo a la regla de un solo smax).
- **`too_large` por par** sobre `SW_MAX_TRACE_CELLS` (2^26 celdas de dirección);
  ni crash ni score fabricado.
- Score-only con `pattern_len` que exceda la shared del device (el kernel
  score necesita `2(pattern_len+1)` ints en shared) se rechaza con el límite
  nombrado.
- Validación idéntica a WFA: punteros nulos con longitud no-cero, longitudes
  negativas, parámetros de scoring inválidos (`match<=0`, `mismatch>=match`,
  `gap_open<=0`, `gap_extend<=0`).

## Rutas de ejecución

- **CPU shim**: ejecuta los kernels reales — `sw_trace_kernel` por llamada
  directa, `sw_score_kernel` bajo emulación de warp (`-pthread`). La API SW es
  gateable en correctitud en CPU, propiedad que el path WFA del shim no tiene
  (WFA en shim sigue siendo shape-only y el test lo dice explícitamente).
- **GPU**: packing de buffers, allocs device, `sw_trace_kernel` si algún par
  pide CIGAR, `sw_score_kernel` en batches puros score-only; launch+sync
  verificados; los CIGARs devueltos se re-validan en la API antes de entregar.

## Evidencia

| verificación | resultado |
|---|---|
| CPU gate completo (scripts/check_kernel_cpu.sh) | verde, incl. `test_api` bajo shim: ejemplo documentado score=12/`6M` span t[3..8]p[0..5]; respuestas conocidas (idéntica 16, gap afín 17 `8M6I8M`, disjuntas 0, vacías, score-only); **120/120** pares vs referencia CPU independiente, CIGARs bien formados; batch heterogéneo; `too_large` en 8192×8193; todos los rechazos |
| Ejemplo público SW | compila y corre bajo el shim: localiza el bloque conservado de 80 nt en mtDNA real (score=155, span `[10..89]` en ambas) |
| **GPU MI210** (job 29227600) | gate completo en el nodo, stage API con hipcc: **RESULT: PASS — API verified on the GPU path against the CPU DP**, incluida toda la batería SW en device |
| **SeqAn3** | **150 filas, 0 discrepancias de score** sobre los resultados que la propia API emitió en device (`--emit-sw`) |

## Defecto latente que el gate GPU sacó a la luz (no es de Fase C)

`smax ≥ ~89` con `with_cigar` pide `(smax+1)(2smax+3)×4B` de shared por bloque
al trace kernel de WFA — sobre 64 KiB (MI210) eso es `smax ≤ 88`. Las secciones
`smax=200/511` de `test_real_sequences`, `test_batch_semantics` y la sección
`smax=128` de `test_api` quedaron obsoletas silenciosamente desde B5 (que hizo
que el trace kernel realmente usara la shared que pide): nadie había vuelto a
correr el stage 6 en un nodo GPU. Esas secciones verifican semántica
resolved/unresolved, que el score kernel responde — ahora van score-only con el
motivo escrito en el test. La cobertura de CIGAR en datos reales sigue en las
secciones `smax=64`.

Esto también pone una fecha de caducidad honesta sobre afirmaciones anteriores:
la documentación de WFA debe leerse sabiendo que **traceback por API está
limitado a `smax ≤ ~88` en MI210** (64 KiB shared). Score-only no tiene ese
límite.

## Lo que NO se verificó aquí

- NVIDIA: el path SW nunca corrió en hardware CUDA. "Portable" sigue sin
  poder afirmarse.
- Rendimiento SW: ninguna medición. Fase D.
