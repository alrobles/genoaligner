# B6 — Throughput con barras de error

> **Medido:** 2026-09-12 · jobs **29226478** (MI210), **29226479** (A100),
> **29226480** (PRO 6000), **29226535** (diagnóstico de la inestabilidad)
> **Métrica:** kernel-only TCUPS = celdas de pares RESUELTOS / tiempo del kernel.
> **Diseño:** todo dentro de UN job por GPU. 7 repeticiones independientes por régimen,
> ambos kernels intercalados A,B,A,B para que la deriva intra-job afecte a los dos.
> **Todos los números de este documento están copiados de los logs**, no de memoria.

---

## 1. Por qué este documento existe y qué NO afirma

Fase 6 publicó números únicos medidos en jobs **separados**. Fase 7 luego midió 3-4x de
varianza entre nodos con la misma entrada, lo que hace que cualquier comparación
cruzando jobs sea aritmética sobre una base móvil (un "1.93x" se publicó y se retractó
por eso). B6 mide dentro de un job y reporta **media, sd, min, max y dispersión**.

Se afirma: **rangos de throughput con su dispersión, por GPU y régimen.**
No se afirma: comparaciones cruzando jobs, ni que la dispersión sea solo ruido — ver §3,
donde resulta ser una propiedad del kernel.

## 2. MI210 (job 29226478)

| régimen | default (mean ± sd) | flat (mean ± sd) | dispersión def. | dispersión flat |
|---|---|---|---|---|
| len=256 smax=64 id=90% | 0.389 ± 0.000 | 0.580 ± 0.002 | 0.3% | 1.2% |
| len=1024 smax=256 id=90% | 0.421 ± 0.000 | 1.121 ± 0.002 | 0.0% | 0.5% |
| len=1024 smax=256 id=70% | 0.472 ± 0.000 | 1.108 ± 0.003 | 0.0% | 0.8% |
| len=2048 smax=128 id=95% | **0.947 ± 0.370** | **0.965 ± 0.404** | **277.8%** | **243.9%** |

Los tres primeros regímenes son **estables al 1.2%**. El cuarto es inestable en ambos
kernels, con valores **discretos** (no una gaussiana): 1.209 repetido, y saltos a 0.320,
0.915, 0.558.

## 3. La inestabilidad NO es ruido — es el kernel, con trabajo idéntico

Diagnóstico dedicado (job 29226535), el régimen inestable, 7 corridas, **mismo nodo,
mismo job**:

    rep    TCUPS  dist_mean  dist_max  wf_off  wf_used  resolved  abandoned
      1    1.210       74.8        85     128     74.8       300          0
      2    1.209       74.8        85     128     74.8       300          0
      3    0.492       74.8        85     128     74.8       300          0
      4    0.894       74.8        85     128     74.8       300          0
      5    0.914       74.8        85     128     74.8       300          0
      6    1.210       74.8        85     128     74.8       300          0
      7    1.209       74.8        85     128     74.8       300          0

**El kernel hace exactamente el mismo trabajo en las 7 corridas** — idéntica distancia,
idénticos wavefronts caminados, idéntico conteo resuelto — y el tiempo varía **2.5x**.
No es medición, ni abandonos, ni el nodo: es el kernel `default` siendo temporalmente
**no determinista** cuando `blockDim = 2·smax+1` (hasta 256 hilos, la mayoría ociosos
esperando en `__syncthreads()`). El `flat` sobre el mismo régimen es estable al 0.2%
(1.260-1.263).

**Consecuencia para cualquier benchmark:** un número único del kernel `default` en un
régimen así es una muestra de una distribución de 2.5x de ancho.

## 4. A100 y PRO 6000 (jobs 29226479, 29226480)

    A100 (job 29226479)
    régimen                   default            flat            flat/default
    len=256 smax=64 id90%      0.528 ± 0.076      0.759 ± 0.081    1.44x   (37%)
    len=1024 smax=256 id90%    0.473 ± 0.038      1.344 ± 0.167    2.84x   (26-34%)
    len=1024 smax=256 id70%    0.486 ± 0.021      1.187 ± 0.060    2.44x   (10%)
    len=2048 smax=128 id95%    1.835 ± 0.180      1.750 ± 0.176    0.95x   (32%)

    PRO 6000 (job 29226480) — dispersión 0.2-0.5% en TODOS los regímenes
    régimen                   default            flat            flat/default
    len=256 smax=64 id90%      1.629 ± 0.002      2.579 ± 0.003    1.58x
    len=1024 smax=256 id90%    1.208 ± 0.001      5.316 ± 0.004    4.40x
    len=1024 smax=256 id70%    1.314 ± 0.001      5.455 ± 0.003    4.15x
    len=2048 smax=128 id95%    4.333 ± 0.003      4.387 ± 0.004    1.01x

**PRO 6000 es el más estable** (0.2-0.5% en todo) y donde el `flat` gana más (hasta
4.40x en el régimen de Fase 6). **A100 quedó inestable en todos los regímenes**
(10-37%), lo que apunta a un nodo ocupado — indistinguible de una propiedad del kernel
con estos datos, así que **no se atribuye**.

## 5. Lo que estos números permiten afirmar

- **Kernel-only TCUPS por GPU**, con dispersión, donde todo resuelve:
  `default`: MI210 0.39-0.47 · A100 0.47-0.53 · PRO 6000 1.21-1.63
  `flat`:    MI210 0.58-1.12 · A100 0.76-1.34 · PRO 6000 2.58-5.46
- **El `flat` es 1.4-4.4x más rápido que el `default`**, consistente con Fase 7.
- **El `default` tiene un modo no determinista de hasta 2.5x con trabajo idéntico**
  (§3), del que el `flat` carece.

## 6. Lo que NO se puede afirmar

- **Nada cruzando jobs.** Cada fila viene de un job distinto.
- **La comparación con Accelign** (9-16 TCUPS en PRO 6000) se reporta como **referencia
  publicada**, no como comparación propia: no se midió su binario con este diseño.
  El orden de magnitud se mantiene (~2-3x por debajo con `default`).
- **El régimen len=2048 no sirve para publicar throughput**: inestable (§3), causa
  identificada pero no arreglada.
- **El resultado de A100**: no se atribuye a la GPU ni al kernel.

## 7. Reglas que salieron de este trabajo

1. **`stats` y el parseo se probaron contra entradas conocidas antes de lanzar.** Un
   helper de estadística con un bug produce números falsos con apariencia de rigor.
2. **Una muestra corrupta se rechaza, no se promedia.** El primer harness capturó un
   valor a la mitad en un rep (0.212 vs 0.421×6); la aritmética lo identificó exacto:
   `(0.421*6+0.212)/7 = 0.3911`, la "media" impresa. Ahora cada corrida va a su archivo,
   el parseo exige exactamente una línea de cada tipo, y rechaza si no resuelven todos
   los pares.
3. **Extraer campos por patrón, no por posición.** Un `sed 's/.*: *//'` greedy tomaba el
   último campo (`reps`) en vez del primero (`pairs`) y rechazó las 28 corridas —
   correctamente, por la razón equivocada.
4. **Cuando un régimen es inestable, medir las cantidades que lo deciden** (distancia,
   wavefronts, resueltos) **en vez de promediar.** Aquí eso distinguió "el kernel hace
   otro trabajo" de "tarda distinto haciendo el mismo" — y resultó ser la segunda, que
   es una propiedad del kernel.
5. **Citar números de los logs, no de la lectura anterior.** La primera versión de este
   documento llevaba valores de otro job (MI210 len=256 decía 0.470/0.904 y es
   0.389/0.580). Se reescribió copiando cada cifra del log correspondiente.
