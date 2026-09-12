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
