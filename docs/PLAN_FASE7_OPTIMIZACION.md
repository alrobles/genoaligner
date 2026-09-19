# Fase 7 — Plan de optimización y siguientes pasos

> **Estado:** diagnóstico COMPLETO y medido. Primera optimización implementada y
> **verificada en GPU** (job 29210729): 1.06-1.65x. El residuo de escalado con
> `smax` está en atribución (job 29210731, ocupación).
> **Fecha:** 2026-09-11 · **Base:** `docs/BENCHMARK_FASE6.md`

---

## 1. Punto de partida (medido, no supuesto)

Línea base de Fase 6, verificada contra el DP de CPU en cada régimen:

    kernel-only (TCUPS)   MI210  RTX6000  A100   V100   L40    PRO6000
    len=1024 smax=256 90% 0.419   0.316  0.705  0.530  1.014   1.208

Y la comparación que importa, **en el mismo hardware**:

    Accelign  (RTX PRO 6000) ...... 9-16 TCUPS
    genoaligner (RTX PRO 6000) .... 1.21-1.63 TCUPS      -> ~6-13x por debajo

---

## 2. Diagnóstico: dónde se va el tiempo (MEDIDO)

Tres experimentos, cada uno más específico que el anterior. Los dos primeros
descartaron hipótesis; el tercero encontró el mecanismo.

### 2.1 El barrido de smax a distancia real constante (job 29210725)

    len=1024, ident=98%, distancia real ~15.2 IDÉNTICA en las 4 corridas:
      smax= 32  block=128   0.966 ms
      smax= 64  block=256   2.190 ms
      smax=128  block=512   4.835 ms
      smax=256  block=1024 10.349 ms

El trabajo por par es el MISMO, pero el tiempo crece ×10.7. La primera lectura
—"está gastando el tiempo en wavefronts vacíos"— era **una hipótesis**.

### 2.2 ¿Cuántos wavefronts camina de verdad? (`bench/h7_exit_probe.cpp`, sin GPU)

    smax  dist_mean  wf_mean  wf/dist
      32       14.9     14.9     1.00
      64       14.9     14.9     1.00
     128       14.9     14.9     1.00
     256       14.9     14.9     1.00     (400/400 correctos)

**Hipótesis refutada.** El early return SÍ dispara: el kernel camina ~distancia
wavefronts en todos los smax. No desperdicia wavefronts.

### 2.3 El mecanismo real: ocupación

El tiempo escala con el número de WARPS del bloque:

    smax= 32 block=128 warps= 4 -> 0.97 ms
    smax= 64 block=256 warps= 8 -> 2.19 ms    (×2.3)
    smax=128 block=512 warps=16 -> 4.84 ms    (×5.0)
    smax=256 block=1024 warps=32 -> 10.35 ms  (×10.7)

`blockDim = 2*smax+1` lanza hasta 1024 hilos, pero los wavefronts caminados
necesitan a lo sumo `2*15+1 = 31`. **76-97% del bloque está ocioso, y el
`__syncthreads()` se paga sobre warps que no hacen nada.** El bucle es mínimo;
el tamaño de bloque es el defecto.

---

## 3. La optimización implementada (`wfa_score_flat.hip`)

**Desacoplar `blockDim` de `smax`.** Un bloque fijo recorre las diagonales del
wavefront con grid-stride interior:

    for (int t = threadIdx.x; t <= 2*s; t += blockDim.x)  k = t - s;

Álgebra, layout, centinelas y test de terminación **sin cambios** — es la misma
recurrencia sobre los mismos slots, con los hilos reasignados. Deliberado: un
cambio de layout invalidaría la evidencia de paridad, y el defecto es ocupación.

Está en **archivo separado** para no tocar el kernel validado: si no gana, se
descarta sin daño.

### Verificación

**Gate CPU** (`bench/h7_flat_parity.cpp`): **PASS** — 1225/1230 resueltos,
**0 desacuerdos** contra el DP de CPU.

**Gate GPU (job 29210729)**: remapping de hilos verificado — **500/500 contra el DP
de CPU** en bloques 64, 128, 256 y 512. Ese es el chequeo que el gate CPU no puede
hacer (a blockDim=1 el grid-stride interior degenera).

### Resultado medido (job 29210729, MI210)

Mismo régimen que la línea base (len=1024, ident=98%, distancia real ~15.2):

    smax   default     flat(128)   mejora
      32   0.966 ms   0.912 ms    1.06x
      64   2.190 ms   1.794 ms    1.22x
     128   4.835 ms   3.248 ms    1.49x
     256  10.349 ms   6.286 ms    1.65x

La mejora **crece con smax**, que es la firma del defecto: cuanto más
desproporcionado `2*smax+1` respecto a las diagonales usadas, más gana el bloque fijo.

**Y el tamaño de bloque dejó de importar:** 64/128/256/512 dan todos 2.48-2.55 ms
(0.205-0.211 TCUPS) sobre la misma entrada. El bloque ya no es el cuello — que es
justo lo que la desacoplación buscaba.

### EL HALLAZGO REAL (job 29210734): el flat no solo es más rápido — es determinista

Mismo régimen repetido 5× **en un solo proceso** (len=1024, 2000 pares, ident=98%):

    flat (block=128)          rep1    rep2    rep3    rep4    rep5      dispersión
      smax= 64              1.794   1.771   1.781   1.771   1.774      ±0.5%
      smax=128              3.229   3.228   3.226   3.230   3.234      ±0.1%
      smax=256              5.329   5.332   5.334   5.335   5.323      ±0.1%

    default (block=2*smax+1)  rep1    rep2    rep3                      dispersión
      smax= 64              8.722   2.871   5.470                      **3.0x**
      smax=128              5.670   9.444   5.653                      **1.7x**
      smax=256             15.766  10.295  10.287                      **1.5x**

**El kernel original varía hasta 3x entre repeticiones del MISMO proceso.** Eso
invalida parcialmente la línea base con la que trabajé:

- los "0.966 / 2.190 / 4.835 / 10.349 ms" de la atribución son promedios de un
  kernel que oscila; el "escalado limpio con smax" que intenté explicar durante
  cuatro hipótesis era **en gran parte varianza del original**, no una propiedad
  del bucle;
- cualquier afirmación de speedup contra un baseline ruidoso es débil por
  construcción.

**Mecanismo:** `blockDim = 2·smax+1` deja 76-97% de los hilos ociosos esperando en
la barrera. Los warps ociosos se programan de forma no determinista y cada
`__syncthreads()` sobre ellos cuesta lo que al planificador le toque. El flat, con
128 hilos y trabajo balanceado, no tiene ese grado de libertad.

**Consecuencia:** la comparación honesta no es "1.93x", es **"1.9x más rápido y
estable frente a uno que no es reproducible"**. Para un benchmark, la
reproducibilidad vale tanto como la velocidad — y esto es lo que hay que reportar.

### Qué se puede afirmar hoy, y qué no

**Se puede afirmar** (medido en GPU, 500/500 verificado vs DP de CPU):
- el flat es **1.06-1.93x más rápido** que el kernel actual;
- el flat es **reproducible al 0.1%**; el actual varía **hasta 3x** intra-proceso;
- el tamaño de bloque dejó de importar (64..512 dan lo mismo), que era el objetivo;
- el remapping de hilos es correcto (500/500 en los 4 bloques).

**No se puede afirmar:**
- que el flat deba fusionarse ya: es un candidato, y el residuo de escalado con
  smax del flat (que sigue existiendo: 1.77 → 5.33 ms de smax 64 a 256) no está
  explicado. Pero ahora se mide contra un baseline reproducible, así que el residuo
  es una propiedad **real** del flat y ya no puede ser varianza.
- ninguna causa del residuo: cuatro hipótesis cayeron (wavefronts vacíos, tamaño de
  bloque, ocupación, zeroing).

### Lección de método de esta fase

Cinco números parecían resultados y no lo eran. Los cuatro primeros fueron
inferencias mías refutadas al medir (wavefronts vacíos, block, smem, ocupación). El
quinto es el más importante: **comparaba contra un baseline que no era
reproducible.** La regla que faltaba —y que ahora queda en el skill— es que
**antes de atribuir una diferencia hay que medir el piso de ruido de ambos lados**;
un baseline que varía 3x no puede sostener un speedup de 1.9x.

### Un error de mi propio gate, corregido

La primera versión comparaba el flat contra el kernel validado **bajo el shim**, y
dio 1087/1230 "fallos". Era inválido: con blockDim=1, el guard
`if (threadIdx.x <= 2*s)` del kernel validado computa UNA sola diagonal, así que
ese run es degenerado y no sirve de referencia (resuelve 138/1230). El flat era el
correcto. Ahora el gate compara contra el DP de CPU, que es la verdad independiente.

---

## 4. Siguientes pasos, en orden

### Paso 0 — ANTES DE CUALQUIER COMPARACIÓN: reportar el piso de ruido

El job 29210734 mostró que el kernel actual varía **hasta 3x intra-proceso**. Toda
comparación futura debe reportar `N repeticiones + dispersión` de AMBOS lados. Un
número único de un kernel ruidoso no es una medición. Esto va primero porque
invalida cualquier speedup calculado sin él — incluido el "1.93x" si se citara solo.

### Paso 1 — Cerrar la verificación del flat

Hecho: gate CPU PASS (0 desacuerdos), gate GPU PASS (500/500 en 4 bloques).
Falta: el residuo de escalado del flat (1.77 → 5.33 ms de smax 64→256) — real,
ahora medible contra baseline estable.

### Paso 2 — Repetir la matriz de Fase 6 con el flat, con dispersión

- Re-correr los regímenes de Fase 6 con el kernel flat en MI210 y PRO 6000,
  **reportando media y dispersión de N repeticiones**.
- Repetir el gate externo (edlib) sobre la salida del flat — misma disciplina que
  Fase 4: el brazo externo corre sobre el kernel que se reporta.
- Actualizar `docs/BENCHMARK_FASE6.md` con antes/después y las barras de error.

### Paso 3 — Las siguientes palancas, por orden de beneficio/riesgo medido

1. **Múltiples pares por bloque.** 8 pares por bloque (1 warp cada uno) llena el
   bloque y amortiza barreras sobre trabajo útil. Requiere A/B por par en shared.
2. **Extensión vectorizada.** `wfa_extend` compara byte a byte. Solo relevante si la
   extensión domina, y **no se ha medido que domine** — medirlo antes.
3. **Reducir barreras.** El kernel validado tiene 3 `__syncthreads()` por wavefront;
   una parece no-op pero **NO lo es** (mi primer intento de quitarla falló el gate).
4. **`smax` adaptativo** = `min(smax, m+n)`. Menor impacto.

Cada palanca: **medir el cuello primero, cambiar una cosa, gate CPU, gate GPU, y
solo entonces creer el número.**

### Paso 4 — Fase 7 real: Smith-Waterman

El masterplan define Fase 7 como el segundo método (SW antidiagonal). Las
optimizaciones de arriba benefician a ambos kernels, así que conviene cerrarlas
antes de escribir el segundo.

---

## 5. Lo que NO se debe reportar como hecho

- Ningún speedup. La optimización **no está verificada en GPU todavía**.
- El flat kernel no está fusionado: es un candidato hasta que el job 29210729 pase.
- La extensión vectorizada no se ha medido: **no se sabe si la extensión domina**.
- `smax > 511` sigue sin ser alcanzable con este mapeo.

---

## 6. Disciplina que este trabajo reforzó

Tres veces en el proyecto un número "obvio" resultó falso, y las tres el remedio
fue el mismo — hacer que la herramienta se niegue a reportarlo:

| caso | el número que parecía resultado | lo que era |
|---|---|---|
| Fase 6 | 136 TCUPS | shared memory = 0: medía el guard |
| Fase 6 | "memset de 475 ms" | bring-up del contexto HIP |
| Fase 7 | "gasta 88-98% en wavefronts vacíos" | el return sí dispara; era ocupación |

Más un cuarto, inverso, en esta misma fase: **mi gate de paridad estaba mal, no el
kernel** (1087 "fallos" de un test inválido). El patrón es constante: cuando un
número sorprende, el primer sospechoso es la medición, no el mundo.
