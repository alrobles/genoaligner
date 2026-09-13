# Plan — Smith-Waterman en genoaligner

> **Estado al escribir esto (2026-09-12):** la referencia SW y su gate están verdes; el
> kernel `sw_score_kernel` **corre correcto en GPU** para `blockDim=1`; dos bugs de GPU
> ya encontrados y arreglados (punteros de host en la struct del device; solape de shared
> estática/dinámica). La **paralelización real no está hecha**: hoy es 1 hilo por bloque,
> correcto y lento.
>
> Jobs: 29226683 (triaje), 29226704 (kernel correcto en GPU), 29226705 (test principal).
>
> **Actualización (2026-09-12, mismo día):** Fase A implementada Y VERIFICADA EN GPU —
> el kernel es warp-per-row con la dependencia intra-fila resuelta como scan de
> prefijos (ver §A1 nota). Gate CPU verde en las dos anchuras (110/110 a warpSize
> 32 y 64, con control negativo). En MI210: 210/210 contra la referencia en 3 jobs,
> y determinismo dev-time 0.2% en nodo exclusivo — docs/RESULTADO_H8_SW1_WARP.md
> tiene los números y el hallazgo de que este cluster comparte GPUs por shards
> (los benches wall-clock miden al vecino). Falta solo el speedup intercalado.

---

## 0. Dónde estamos y qué falta, en una frase

SW **funciona y es correcto** en la configuración más simple posible; falta **hacerlo
rápido sin romperlo**, y falta el **traceback**. Esa es toda la fase.

---

## 1. Lo que ya está hecho (no requiere trabajo)

| pieza | estado | evidencia |
|---|---|---|
| Referencia O(mn) en CPU | ✅ | `tests/sw/test_sw_reference.cpp`, 15 casos, expectativas a mano |
| Propiedad local-vs-global **aserida** | ✅ | local 12 vs global −8 sobre el mismo par |
| Gate de paridad kernel↔referencia | ✅ | `tests/sw/test_sw_parity.cpp`, 6/6 |
| Kernel correcto en GPU (blockDim=1) | ✅ | job 29226704: 9 casos, incluido 400×160 → 121 |
| Triaje por bisección | ✅ | `tests/sw/test_sw_dbg.cpp` — nombró el caso culpable en una corrida |

**Dos bugs de GPU ya cazados, ambos invisibles al gate CPU:**
1. **Punteros de host en la struct del device.** `SWPairView` guarda `const char*`; el
   test copiaba la struct con `hipMemcpy`, copiando los punteros. El fault fue en
   `0x7ffe...` — la pila del host. Falla con m=1,n=1: estructural. *Este proyecto ya lo
   había documentado y arreglado para WFA; lo repetí.*
2. **Shared estática y dinámica se solapan.** `__shared__ int r[SW_BLOCK]` junto a
   `extern __shared__ int smem[]`: la dinámica empieza *después* de las estáticas, así que
   `smem[0]` dejó de ser la base y los buffers chocaron.

---

## 2. Fase A — Paralelizar sin romper la correctitud

**El problema, concretamente.** El kernel camina fila por fila y lleva `diag_prev` por
hilo. Con un grid-stride sobre columnas, `H[j-1]` y `E[j-1]` los escribe **otro hilo** de
la misma fila, y una barrera por fila **no** garantiza esa visibilidad. Por eso hoy va con
1 hilo por bloque.

**Tres opciones, con su coste y su riesgo:**

| opción | idea | coste | riesgo |
|---|---|---|---|
| **A1. Una fila por warp, sin stride** | cada hilo de un warp toma una columna contigua; las dependencias intra-fila se resuelven con `__shfl_up` (la columna j−1 está en la lane anterior) | medio | el `shfl` tiene que respetar `j-1` cuando el hilo anterior tiene la columna anterior: si el patrón no es lane↔columna exacto, no sirve |
| **A2. Anti-diagonal con UNA barrera por diagonal** | el estándar de la literatura, paralelismo máximo | alto | **ya medimos que este patrón da tiempos no deterministas** (B6: 2.5x con trabajo idéntico en el kernel WFA). Adoptarlo reintroduce el defecto |
| **A3. Filas en paralelo (wavefront de filas)** | varias filas en vuelo a la vez, cada una por un bloque distinto | medio | más memoria (varias filas vivas) y sincronización entre bloques por cooperación |

**Recomendación: A1, medida contra A3.** Razón: A2 está descartada por medición previa
del propio proyecto, y A1 ataca la dependencia real (intra-fila) sin barreras extra.

> **Nota A1 — lo que el `shfl_up` directo NO puede hacer (2026-09-12).** El riesgo
> anotado arriba resultó real: `H[i][j]`/`E[i][j]` necesitan el valor de la misma fila
> que la lane vecina está calculando *en el mismo paso* — en lockstep ese valor aún no
> existe cuando se pide. La forma correcta de A1 es reformular E como scan de prefijos:
> `c[j] = max(0, diag+s, F)`, `U[j] = max(U[j-1]+d, c[j]+ge·j)` con `d=max(0,ge−go)`, y
> `E[j] = U[j-1] − go − ge·(j−1)`. El álgebra completa está en `sw_kernel.hip` y fue
> verificada contra la recurrencia directa en 6000 casos aleatorios antes de escribir
> el kernel. Resultado: cero barreras (ni `__syncthreads` ni `__syncwarp`), cada lane
> solo toca slots de shared que le pertenecen, todo lo demás viaja por shuffles.
> El shim CPU aprendió a emular warps con threads reales (`shim::run_block`), así que
> `blockDim>1` ahora está gateado en CPU en segundos — a ambas anchuras (32 y 64).

**Criterio de éxito de la Fase A (todo obligatorio):**
1. Paridad **exacta** contra la referencia en los 15 casos, con `blockDim > 1`.
   → ✅ CPU emulada: 110/110 a warpSize 32 y 64. ✅ GPU: 210/210 en MI210,
   `block=256` (4 warps × 64), 3 jobs.
2. La paridad se rompe si se quita la sincronización (test que falla con el bug).
   → ✅ control negativo integrado: con el exchange desactivado fallan 100/110.
3. **Determinismo**: 7 corridas de la misma entrada dan tiempos con dispersión < 5%
   (el estándar que B6 estableció, porque el kernel WFA falló ahí).
   → ✅ dev-time (hipEvent) spread = 0.2% en nodo exclusivo (job 29227117).
   En nodo compartido los stalls del vecino llegan a 400x — ver
   RESULTADO_H8_SW1_WARP.md §3: las GPUs se comparten por shards.
4. Speedup medido con el **diseño intercalado** de B6, no cruzando jobs.
   → ✅ 4.93x (46.4ms → 9.4ms) sobre el kernel de 1 hilo, job 29227132,
   nodo exclusivo, ambos kernels estables (<1.5% spread). Número honesto:
   el batch es de pares cortos; con patrones largos sería mayor.

---

## 3. Fase B — Traceback (el CIGAR)  ✅ COMPLETA

**Por qué va después y por qué es la parte peligrosa.** En WFA, el camino del CIGAR
**nunca funcionó por la API y su test pasaba** — los pares del test eran tan cortos que
un tamaño de shared equivocado era aceptado por casualidad. Aquí no se repite: el
traceback SW tendrá su propio gate **contra el CIGAR de SeqAn3**, no solo contra el score.

**Diseño implementado (2026-09-12, commit 7640de6):** `sw_trace_kernel` — un hilo por
par, pase forward serial que graba **un byte de dirección por celda** (hsrc/esrc/fsrc),
walk hacia atrás leyendo solo la tabla. Workspace `(m+1)(n+1)` bytes/par con límite
declarado (`SW_TRACE_TOO_BIG`) — decisión de memoria tomada explícitamente, como pedía
el plan. Detalles y evidencia: `docs/RESULTADO_H9_SW2_TRACE.md`.

**Criterio de éxito:**
1. El CIGAR reconstruye las secuencias (consume exactamente los tramos alineados).
   → ✅ well-formed sobre el tramo alineado, 113/113 × 2 esquemas CPU, 209/209 GPU.
2. Re-puntuado: `score(CIGAR) == score` del kernel.
   → ✅ re-score exacto en todos los casos; además CIGAR idéntico al walk
   independiente por valores.
3. **Comparado contra SeqAn3** en ambos backends — el oráculo externo, no el propio.
   → ✅ SeqAn3 local+afín: 201/201 filas del job GPU, 0 discrepancias
   (job 29227484). "Ambos backends" (NVIDIA) queda pendiente con la portabilidad.
4. Los empates se manejan: distintos CIGARs pueden ser óptimos, así que se compara
   **score**, nunca la cadena de operaciones.
   → ✅ el gate externo compara score; los spans difieren solo en empates legítimos.

**Hallazgo documentado:** `gap_extend > gap_open` es degenerado para traceback (la DP
re-abre gaps; el CIGAR los fusiona y el re-score diverge por construcción). El score
sigue siendo exacto; Fase C lo rechaza explícitamente.

---

## 4. Fase C — Integrar en la API pública ✅ (commit b25bf81, job 29227600)

Decisión tomada: **superficie separada**, no un campo `algorithm` —
`SWRequest`/`SWAlignResult`/`SWBatchResult` + `align_sw`/`align_sw_batch`.
`AlignResult.score` (distancia) y `SWAlignResult.score` (puntuación) no pueden
confundirse porque son tipos distintos; `smax` no existe en `SWRequest`.

Verificado (docs/RESULTADO_H10_SW_API.md): CPU gate verde con correctitud SW
real bajo el shim; MI210 job 29227600 — API test con hipcc PASS incl. SW en
device; SeqAn3 150/150 sobre los resultados emitidos por la API en device;
`examples/align_sw.cpp` compila y corre. Hallazgo colateral documentado:
traceback WFA por API está limitado a `smax ≤ ~88` en MI210 por shared memory.

NVIDIA verificado después (docs/RESULTADO_H11_SW_CUDA.md, job 29230631):
V100 warpSize=32 real — score 210/210, trace 209/209, API PASS, SeqAn3
201/201 + 150/150. SW corre en ambos vendors desde el mismo árbol.

Queda en la deuda honesta: rendimiento no medido (Fase D).

---

## 5. Fase D — Rendimiento y evidencia

- Medir con el diseño de B6 (intercalado, dentro de un job, media + dispersión).
- Comparar con **parasail** (CPU, SIMD) y, si corre, con la implementación CUDA de
  referencia — declarando **cuál** se midió y cuál se cita.
- **No prometer nada hasta tener los números.** Un SW correcto y lento es un resultado
  válido y publicable; un SW rápido y mal medido, no.

---

## 6. Riesgos, en orden de probabilidad

| riesgo | por qué | mitigación |
|---|---|---|
| Reintroducir el modo no determinista | A2 (anti-diagonal) es el diseño "estándar" y es el que falla aquí | no usar A2 sin medir; criterio de determinismo explícito |
| El traceback repite el historial de WFA | ya pasó: CIGAR roto con test verde | gate contra SeqAn3 desde el primer commit de traceback |
| Memoria O(mn) | SW no tiene la estructura de banda de WFA | decidir el límite y **documentarlo**, no descubrirlo en producción |
| Cambio de semántica en `AlignResult.score` | distancia vs puntuación en el mismo campo | campo/API distinto, documentado |
| Un tercer bug solo-GPU | van dos, ambos invisibles al shim | correr en GPU **en cada paso**, no al final |

---

## 7. Orden de trabajo, con su verificación

| # | tarea | verificación | esfuerzo |
|---|---|---|---|
| 1 | Commitear el kernel GPU-correcto (blockDim=1) como base | job 29226705 PASS | ya |
| 2 | A1: un warp por fila con `__shfl_up` | paridad con blockDim>1 + test que falla sin sync | 1 sesión |
| 3 | Determinismo de A1 | 7 corridas, dispersión < 5% | 1 job |
| 4 | Speedup de A1 | diseño intercalado de B6 | 1 job |
| 5 | Traceback + gate SeqAn3 | score y CIGAR contra el oráculo | 2 sesiones |
| 6 | API: `algorithm` + params + docs | ejemplo del README intacto; ejemplo SW nuevo | 1 sesión |
| 7 | Bench y comparación | números con dispersión | 1 job |
| 8 | Cerrar el plan: docs, README, MASTERPLAN | revisión | — |

**La regla que gobierna todo esto**, y que salió de esta misma fase: *correr en GPU en
cada paso, no al final.* Los dos bugs de este kernel eran invisibles al gate CPU y
aparecieron en la primera corrida real. Un plan que dejara la verificación GPU para el
final habría acumulado varias capas de suposiciones sobre una base rota.

---

## 8. Lo que este plan NO promete

- **No promete** que SW sea rápido. Hoy es correcto y lento, y mejorar eso es la Fase A.
- **No promete** que SW sea competitivo con parasail. Se medirá y se dirá.
- **No promete** traceback para genomas grandes: el límite de memoria se decide en Fase B.
- **No promete** que el modo no determinista no vuelva. Por eso el criterio 3 de la Fase A
  es explícito y se mide.
