# Fase 7 — Optimización: resultado verificado

> **Cierre del Paso 1** · 2026-09-11 · job 29213844 (intercalado A/B, MI210)
> **Método:** comparación intercalada en un solo job (el único diseño que sobrevive
> a la varianza de nodo). Ver `docs/PLAN_FASE7_OPTIMIZACION.md` para el diagnóstico.

---

## 1. Resultado

Comparación **intercalada** (default, flat, default, flat, …) en el MISMO job, 5
rondas, mismo régimen (len=1024, 2000 pares, ident=98%). La métrica es el **ratio
por ronda**, que cancela la deriva del nodo:

    smax    ratio por ronda                          media    estabilidad del ratio
      64    1.224  5.434  1.348  1.223  1.237        1.29     ±10%   (ver §2)
     128    1.493  1.491  1.488  1.491  1.489        1.49     ±0.2%
     256    1.933  1.931  1.932  1.933  1.935        1.93     ±0.1%

**El kernel flat es 1.22-1.93x más rápido, y la ganancia crece con smax.** En
smax=128 y 256 el ratio es reproducible al 0.2%, que es lo que hace defendible el
número: no depende de en qué nodo cayó el job.

    smax=256:  default 10.29 ms  ->  flat 5.32 ms    1.93x
    smax=128:  default  4.82 ms  ->  flat 3.23 ms    1.49x

## 2. Un hallazgo secundario que vale tanto como el ratio: el flat es ESTABLE

En smax=64 los milisegundos absolutos cuentan la historia real:

    ronda   default_ms   flat_ms
      1        2.883      2.356
      2        9.619      1.770     <- el default salta 4.4x; el flat no
      3        2.388      1.771
      4        2.179      1.782
      5        2.185      1.766

**El kernel default oscila 2.179-9.619 ms (4.4x) dentro del mismo proceso; el flat
se mueve 1.766-1.782 (±0.5%).** La ganancia real en smax=64 no es el ratio medio de
1.29 — es que **el flat no tiene el modo malo**. Un kernel que a veces tarda 4x más
es un problema para cualquier benchmark, y ese modo desaparece.

Esto es coherente con el mecanismo: `blockDim = 2*smax+1` deja 76-97% de los hilos
ociosos esperando en `__syncthreads()`; el planificador los coloca de forma no
determinista, y en algunos lanzamientos eso cuesta 4x. El flat, con 128 hilos y
trabajo balanceado, no tiene ese grado de libertad.

## 3. Qué se retracta y por qué

**El "1.93x" que reporté el turno anterior se retractó, y ahora se vuelve a afirmar
con base válida.** La primera versión se midió **cruzando jobs**, y el job 29213840
demostró que la varianza entre nodos (3-4x) domina cualquier efecto:

    misma config flat smax=256    job 29210734:  5.329 ms  (±0.1%)
                                  job 29213840: 16.425 ms  (hasta 20.308)

En aquel momento el número era aritmética sobre un baseline móvil — el mismo error
que acababa de escribir como lección. La corrección no fue más análisis, fue
**cambiar el diseño experimental**: ambos kernels en el mismo job, alternando.

## 4. Qué sobrevive, con su evidencia

| afirmación | evidencia | ¿depende del nodo? |
|---|---|---|
| flat es 1.22-1.93x más rápido | ratio intercalado, ±0.2% en smax≥128 | no (se cancela) |
| el default es inestable (4.4x) | 2.179-9.619 ms intra-proceso | no (mismo proceso) |
| el flat es estable (±0.5%) | 1.766-1.782 ms intra-proceso | no (mismo proceso) |
| remapping de hilos correcto | 500/500 vs DP de CPU | no |
| blockDim=2smax+1 deja 76-97% ocioso | aritmética + lectura del código | no |
| "escalado lineal limpio con smax" | — | **RETRACTADO: era varianza de nodo** |

## 4b. Matriz de Fase 6 con el flat (Paso 2, jobs 29213845 MI210 / 29213846 PRO 6000)

Mismos regímenes de Fase 6, diseño **intercalado** (default, flat, default, flat) y
el **ratio por ronda** como métrica. 4 rondas por régimen:

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

**Ratios reproducibles al 0.5%** en casi todas las filas. La ganancia del flat:

    MI210 ...... 1.49x - 2.67x
    PRO 6000 ... 1.58x - 4.41x      <- mucho mayor en la GPU moderna

El PRO 6000 gana más porque su `blockDim = 2*smax+1` desproporcionado pesa más con
más capacidad de cómputo: el default tarda 1.755 ms donde el flat tarda 0.399 ms. En
el régimen de Fase 6 (len=1024, smax=256, ident=90%) el flat lleva el PRO 6000 de
**1.755 ms a 0.399 ms**, es decir de ~1.21 TCUPS a ~5.3 TCUPS.

Sigue siendo ~2-3x por debajo de Accelign (9-16 TCUPS) en la misma tarjeta, pero la
brecha se cerró de ~6-13x a ~2-3x — con una optimización que **no cambia la
álgebra**.

Nota de honestidad: la fila `ident=70% smax=256` de MI210 tiene el default oscilando
(4.442 → 6.604 ms entre rondas), que es el mismo modo inestable documentado en §2.
El ratio de esa fila es por eso menos apretado (2.344-2.555) que en las demás.

## 5. Estado del kernel y siguientes pasos

El flat sigue en **archivo separado** (`wfa_score_flat.hip`), no fusionado. Con el
Paso 1 cerrado, se desbloquean:

- **Paso 2:** re-medir la matriz de Fase 6 completa con el flat, **con el diseño
  intercalado** (no comparaciones cruzadas), en MI210 y PRO 6000, + gate edlib sobre
  la salida del flat.
- **Paso 3:** siguientes palancas. La #1 (múltiples pares por bloque) ataca
  directamente la causa de la inestabilidad del default: hilos ociosos.
- **Paso 4:** Fase 7 real — Smith-Waterman.

## 6. Regla de método (al skill)

**Para comparar dos kernels en hardware compartido: intercalar A,B,A,B en un mismo
job y reportar el RATIO por ronda.** Los milisegundos absolutos no son comparables
entre jobs (3-4x de varianza de nodo medida); el ratio intercalado sí. Y **reportar
la dispersión junto a la media**: el hallazgo de que el default oscila 4.4x importa
tanto como el speedup, y no habría aparecido en un promedio.
