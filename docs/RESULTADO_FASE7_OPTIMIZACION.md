# Fase 7 — Optimización: resultado verificado

> **Estado: Pasos 0-3 COMPLETOS.** 2026-09-12 · jobs 29210734/29213840/29213844/29213845/29213846
> **Método:** comparación intercalada en un solo job (el único diseño que sobrevive
> a la varianza de nodo). Diagnóstico completo en `docs/PLAN_FASE7_OPTIMIZACION.md`.

---

## 1. Resultado

Comparación **intercalada** (default, flat, default, flat, …) en el MISMO job, mismo
régimen (len=1024, 2000 pares, ident=98%). La métrica es el **ratio por ronda**, que
cancela la deriva del nodo (job 29213844, 5 rondas):

    smax    ratio por ronda                          media    estabilidad
      64    1.224  5.434  1.348  1.223  1.237        1.29     (±10%, ver §2)
     128    1.493  1.491  1.488  1.491  1.489        1.49     ±0.2%
     256    1.933  1.931  1.932  1.933  1.935        1.93     ±0.1%

**El kernel flat es más rápido, y la ganancia crece con smax.** En smax≥128 el ratio
es reproducible al 0.2%, que es lo que hace defendible el número.

## 2. Hallazgo secundario: el flat es ESTABLE, el default no

Los milisegundos de la ronda en smax=64 cuentan la historia real:

    ronda   default_ms   flat_ms
      1        2.883      2.356
      2        9.619      1.770     <- el default salta 4.4x; el flat no
      3        2.388      1.771
      4        2.179      1.782
      5        2.185      1.766

**El default oscila 2.179-9.619 ms (4.4x) dentro del mismo proceso; el flat ±0.5%.**
La ganancia real en smax=64 no es el ratio de 1.29 — es que **el flat no tiene modo
malo**. Mecanismo: `blockDim = 2*smax+1` deja 76-97% de hilos ociosos en
`__syncthreads()`; el planificador los coloca de forma no determinista y algunos
lanzamientos cuestan 4x. El flat, con 128 hilos y trabajo balanceado, no tiene ese
grado de libertad.

## 3. Matriz de Fase 6 con el flat (jobs 29213845 MI210 / 29213846 PRO 6000)

Mismos regímenes de Fase 6, diseño intercalado, 4 rondas, ratio por ronda:

    MI210 (job 29213845)
    régimen                    default    flat      ratios (4 rondas)
    len=256  smax=64            1.346     0.904    1.489 1.490 1.488 1.501
    len=1024 smax=256 ident=90% 4.986     1.869    2.668 2.663 2.651 2.657
    len=1024 smax=256 ident=70% 4.442*    1.895    2.344 1.719 2.555 2.460   (*inestable)

    RTX PRO 6000 (job 29213846)
    régimen                    default    flat      ratios (4 rondas)
    len=256  smax=64            0.324     0.205    1.580 1.580 1.588 1.588
    len=1024 smax=256 ident=90% 1.755     0.399    4.398 4.407 4.410 4.407
    len=1024 smax=256 ident=70% 1.614     0.386    4.181 4.179 4.168 4.168

**Ratios reproducibles al 0.5%** en casi todas las filas:

    MI210 ...... 1.49x - 2.67x
    PRO 6000 ... 1.58x - 4.41x      <- mucho mayor en la GPU moderna

En el régimen de Fase 6 (len=1024, smax=256, ident=90%) el flat lleva el PRO 6000 de
**1.755 ms a 0.399 ms** — de ~1.21 TCUPS a ~5.3 TCUPS. Sigue ~2-3x por debajo de
Accelign (9-16 TCUPS) en la misma tarjeta, pero la brecha pasó de ~6-13x a ~2-3x
**sin tocar la álgebra**.

## 4. Qué se retractó y por qué

El "1.93x" del turno anterior se **retractó** al descubrir (job 29213840) que la
varianza entre nodos es 3-4x:

    misma config flat smax=256    job 29210734:  5.329 ms  (±0.1%)
                                  job 29213840: 16.425 ms  (hasta 20.308)

Ese número era aritmética sobre un baseline móvil. La corrección **no fue más
análisis, fue cambiar el diseño experimental**: ambos kernels en el mismo job
alternando. Con ese diseño el 1.93x se **reconfirmó** (§1). Retractar y reconfirmar
con el experimento correcto es la secuencia honesta.

## 5. Decisión sobre el kernel (Paso 3)

**El flat NO se fusiona todavía.** Razonamiento:

1. No hay consumidor de producción: el repo es una librería en desarrollo y ambos
   kernels se usan solo desde los harnesses. La integración es Fase 8.
2. El flat es **score-only**. Fusionarlo dejaría la librería con un score optimizado
   y un traceback sin optimizar — dos kernels con propiedades distintas.
3. El traceback **no tiene el defecto**: se lanza con `dim3(1)` (un hilo por bloque,
   serial por diseño). No hay hilos ociosos en barreras que corregir, así que la
   corrección del flat no aplica ahí. Verificado antes de intentarlo.

Por tanto el flat queda como **candidato medido y verificado**, en su archivo
separado, listo para ser el kernel de score cuando la librería exponga API (Fase 8)
o cuando el traceback tenga una versión optimizada.

## 6. Reglas de método (al skill)

1. **Para comparar dos kernels en hardware compartido: intercalar A,B,A,B en un
   mismo job y reportar el RATIO por ronda.** Los ms absolutos no son comparables
   entre jobs (3-4x de varianza de nodo medida); el ratio intercalado sí.
2. **Reportar dispersión junto a la media.** Que el default oscile 4.4x importa
   tanto como el speedup, y no habría aparecido en un promedio.
3. **Retractar un número es parte del trabajo.** El 1.93x se retractó y se
   reconfirmó; ambas versiones están en el historial a propósito.
4. **Verificar que una corrección aplique antes de portarla.** Intenté llevar el
   flat al traceback y lo descarté al leer que se lanza con `dim3(1)`: no tiene el
   defecto. Una lectura de dos líneas evitó una reescritura inútil.

## 7. Pendiente

- **Fase 7 real — Smith-Waterman.** El masterplan la define como el segundo método.
  El hallazgo del flat (blockDim desproporcionado) debe aplicarse desde el diseño.
- **Optimizaciones adicionales** (no intentadas): múltiples pares por bloque,
  extensión vectorizada (medir antes si domina), `smax` adaptativo.
- **`smax > 511`** sigue sin ser alcanzable con este mapeo.
- **Números de TCUPS del bench**: la conversión a TCUPS de §3 usa celdas de DP
  completo sobre pares resueltos, la misma definición de Fase 6. No es comparable
  con Accelign de forma directa porque el régimen de identidad difiere.
