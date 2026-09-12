# Fase 7 — Plan de optimización y siguientes pasos

> **Estado:** diagnóstico COMPLETO y medido. Primera optimización implementada,
> gate CPU en verde, verificación GPU **en curso** (job 29210729).
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

### Verificación en curso

- **Gate CPU** (`bench/h7_flat_parity.cpp`): **PASS** — 1225/1230 resueltos,
  **0 desacuerdos** contra el DP de CPU.
- **Job GPU 29210729**: verifica el remapping de hilos real (el gate CPU corre a
  blockDim=1 y NO lo ejercita) + mide el barrido de smax contra la línea base.

### Un error de mi propio gate, corregido

La primera versión comparaba el flat contra el kernel validado **bajo el shim**, y
dio 1087/1230 "fallos". Era inválido: con blockDim=1, el guard
`if (threadIdx.x <= 2*s)` del kernel validado computa UNA sola diagonal, así que
ese run es degenerado y no sirve de referencia (resuelve 138/1230). El flat era el
correcto. Ahora el gate compara contra el DP de CPU, que es la verdad independiente.

---

## 4. Siguientes pasos, en orden

### Paso 1 — Cerrar la verificación del flat (en curso)

Leer job 29210729. Criterio de fusión, todo obligatorio:
- `--verify` con 0 desacuerdos contra el DP de CPU en cada bloque probado
- el barrido de smax plano o mejor que la baseline (0.966/2.190/4.835/10.349 ms)
- elegir `WFA_FLAT_BLOCK` **con el dato del barrido**, no a ojo

### Paso 2 — Si funciona: propagar y re-medir la matriz

- Re-correr los regímenes de Fase 6 con el kernel elegido, en MI210 y PRO 6000.
- Actualizar `docs/BENCHMARK_FASE6.md` con la comparación antes/después.
- Repetir el gate de paridad externo (edlib) sobre la salida del kernel nuevo —
  la misma disciplina que Fase 4: el brazo externo debe correr sobre el kernel
  que se reporta.

### Paso 3 — Las siguientes palancas, por orden de beneficio/riesgo medido

1. **Múltiples pares por bloque.** Un bloque con 8 pares (1 warp cada uno) llena
   el bloque y amortiza las barreras sobre trabajo útil. Requiere un A/B por par
   en shared. Es la continuación natural del hallazgo de ocupación.
2. **Extensión vectorizada.** `wfa_extend` compara byte a byte. Cargar palabras de
   8 bytes y comparar con XOR + find-first-set reduce el bucle de extensión. Solo
   relevante cuando la extensión domina, y **no se ha medido que domine** — hay que
   medirlo antes.
3. **Reducir el número de barreras.** El kernel validado tiene 3 `__syncthreads()`
   por wavefront; una parece no-op pero **NO lo es** (mi primer intento de quitarla
   falló el gate). Analizar cuáles son realmente necesarias, con el gate como red.
4. **`smax` adaptativo.** Hoy se pasa `smax` fijo; la cota natural es
   `min(smax, m+n)`. Menor impacto que las anteriores.

Cada palanca sigue la misma regla: **medir el cuello primero, cambiar una cosa,
gate CPU, gate GPU, y solo entonces creer el número.**

### Paso 4 — Fase 7 real: Smith-Waterman

El masterplan define Fase 7 como el segundo método (SW antidiagonal), reusando la
infraestructura. Las optimizaciones de arriba benefician a ambos kernels, así que
conviene cerrarlas antes de escribir el segundo.

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
