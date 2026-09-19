# RESULTADO H9-SW2 — SW traceback (CIGAR), Fase B

Fecha: 2026-09-12 · Rama: main · Commit base: 7640de6
Job GPU: **29227484** (MI210, r06r18n01, `gres=gpu:mi210:1`)

## Qué se construyó

`sw_trace_kernel` — un hilo por par. El pase forward es la recurrencia serial
directa (camino de código *distinto* del scan warp-per-row, a propósito) que
graba **un byte de dirección por celda**: `hsrc` (stop/diag/E/F), `esrc`
(open/extend), `fsrc` (open/extend). El backward walk lee solo la tabla y emite
el CIGAR en la convención WFA (0=M 1=X 2=I 3=D, reversa, `cigar_meta` de 4
ints). `SWTraceResult` reporta score + coordenadas start/end (0-based).

**Decisión de memoria, explícita** (la exigencia del plan): `(m+1)(n+1)` bytes
de direcciones + `2(n+1)` ints de scratch por par — no `3·mn` ints. El caller
dimensiona `dir_stride`; un par que no cabe reporta `SW_TRACE_TOO_BIG` en vez de
escribir fuera. Es un límite de tamaño soportado y declarado, no traceback a
escala genoma — banda/checkpointing queda como diseño futuro si hace falta.

## Evidencia

| verificación | resultado |
|---|---|
| CPU shim (test_sw_trace) | 113/113 × 2 esquemas ({1,-1,2,1} y {2,-3,3,1}) — trace.score == score-kernel == referencia, CIGAR **idéntico** al walk independiente por valores, well-formed + re-score exacto |
| Control negativo | CIGAR con op volteado: 0 pasan; byte de dirección corrompido: el walk diverge (detectado) |
| ASan/UBSan | test_sw_trace verde en la pasada sanitizer del gate |
| **GPU MI210** (job 29227484) | **209/209** casos: score==kernel==referencia, CIGAR well-formed, re-scored, igual al walk por valores |
| **SeqAn3** (oráculo externo) | selftest 7/7 (incl. el pin del mapeo de gaps); **201 filas, 0 discrepancias de score** sobre los casos emitidos por el job GPU; además 106/106 sobre el TSV del gate CPU |

Mapeo de gaps SeqAn3 verificado por selftest: nuestro `go + ge·(L−1)` ↔
`open_score = −(go−ge)`, `extension_score = −ge` (SeqAn3 cobra `open + ext·L`).

## Hallazgo: `ge > go` es degenerado para TRACEBACK

El gate lo encontró, no lo asumí: con `gap_extend > gap_open` la recurrencia
sigue siendo exacta en score, pero prefiere **re-abrir** gaps de longitud 1
adyacentes en vez de extender (dos opens cuestan `2·go < go+ge`). El CIGAR no
puede representar eso — los ops consecutivos se fusionan en un run cuyo re-score
difiere del score de la DP **por construcción** (ej.: score 14, CIGAR `3M2I5M`
re-puntúa 13 bajo {2,-3,1,2}).

Consecuencia documentada en `sw_kernel.hip`: el score kernel sigue siendo exacto
con `ge>go`; el traceback **no lo soporta**. Fase C rechazará `ge > go` con un
error explícito. Análogo al caso `match=0` del score: la DP es correcta pero el
resultado es degenerado.

## Lo que esto NO cubre todavía

- NVIDIA: el kernel es portátil por construcción (sin shuffles, thread-per-pair)
  pero **no corrido** en CUDA todavía — pendiente del paso de portabilidad.
- Escalas genómicas: límite declarado, no implementado con banda.
- API pública: SW aún no está expuesto (Fase C).
