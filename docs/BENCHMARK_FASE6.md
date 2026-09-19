# Fase 6 — Benchmark honesto

> **Medido:** 2026-09-11 · **Backends:** ROCm 6.4.3/hipcc, CUDA 12.4 y 12.8/nvcc
> **GPUs:** MI210, RTX 6000, A100, V100, L40, RTX PRO 6000 — seis, dos vendors
> **Jobs:** 29210654 (inválido), 29210660/61 (MI210/q6000), 29210662 (A100),
>            29210664 (V100), 29210665 (L40), 29210709 (PRO 6000)
> **Herramienta:** `bench/tcus.cpp` · `scripts/h6_bench{,_cuda,_a100,_v100,_l40,_pro6000}.sbatch`
> **HEAD medido:** 6d27737

## 0. Qué se afirma y qué no

**Se afirma:** el kernel de score WFA corre desde una sola fuente en **seis GPUs de
dos vendors** (MI210, RTX 6000, A100, V100, L40, RTX PRO 6000), y rinde
**0.32-1.63 TCUPS** en los regímenes donde el trabajo se resuelve completo, verificado
contra el DP de CPU sobre una muestra de pares en cada corrida y en cada GPU.

**No se afirma:** que genoaligner sea competitivo con Accelign. Medido **en la misma
GPU** (RTX PRO 6000): Accelign 9-16 TCUPS vs nosotros 1.21-1.63 → seguimos ~6-13x por
debajo (§1b). El objetivo del proyecto es portabilidad, no récord de TCUPS. Tampoco se
afirma que la capa de portabilidad sea gratis: la matriz mide silicio, no la capa (§1a).

## 1. Resultados — matriz de 6 GPUs (jobs 29210660/61/62/64/65/29210709)

`kernel-only` = celdas de los pares RESUELTOS / tiempo del kernel. 1 TCUPS = 1e12
celdas/s. `end-to-end` excluye el warm-up de contexto y la generación de casos (§2, §4).

    kernel-only (TCUPS)
    régimen                  MI210  RTX6000  A100   V100   L40    PRO6000
    len=256,  smax=64,  90%  0.387   0.339  0.446  0.539  1.477   1.629
    len=1024, smax=256, 90%  0.419   0.316  0.705  0.530  1.014   1.208
    len=1024, smax=256, 70%  0.469   0.431  0.725  0.676  1.079   1.314
    len=1024, smax=64,  90%  (inválido: 1945/2000 pares abandonados — ver §3)

    end-to-end (TCUPS)
    régimen                  MI210  RTX6000  A100   V100   L40    PRO6000
    len=256,  smax=64,  90%  0.035   0.105  0.107  0.115  0.213   0.212
    len=1024, smax=256, 90%  0.164   0.208  0.333  0.282  0.499   0.572
    len=1024, smax=256, 70%  0.165    —     0.333  0.320  0.528   0.632

Verificación: **25/25 pares contra el DP de CPU en cada régimen resuelto, en las seis
GPUs** (3/3 en el régimen abandonado, que es todo lo que hay para comparar).

**Lectura de la matriz (ordenada):**

    PRO6000 (Blackwell, sm_120) 1.21-1.63 TCUPS   <- la más rápida
    L40     (Ada, sm_89) ....... 1.01-1.48
    A100    (Ampere, sm_80) .... 0.45-0.73
    V100    (Volta, sm_70) ..... 0.53-0.68
    MI210   (AMD, gfx90a) ...... 0.39-0.47
    RTX6000 (Turing, sm_75) .... 0.32-0.43

El orden sigue el ancho de banda de memoria, no la edad ni el vendor: el V100
(HBM2) adelanta al RTX 6000 y al MI210 pese a ser la arquitectura más vieja. Pero el
L40 duplica al A100 con ancho de banda similar, y el PRO 6000 solo adelanta al L40
por ~1.2-1.6x pese a tener mucho más ancho de banda (GDDR7). Así que **el ancho de
banda ordena la tabla pero no la explica sola**: la arquitectura y las cachés también
pesan. Se reporta como observación, sin atribuir una causa única.

### 1a. Los backends entre sí (H6: ¿la abstracción cuesta rendimiento?)

El MISMO `bench/tcus.cpp`, seis GPUs, dos toolchains (hipcc / nvcc 12.4 y 12.8):

    régimen                  MI210  RTX6000  A100   V100   L40    PRO6000
    len=256,  smax=64, 90%   0.387   0.339  0.446  0.539  1.477   1.629
    len=1024, smax=256,90%   0.419   0.316  0.705  0.530  1.014   1.208
    len=1024, smax=256,70%   0.469   0.431  0.725  0.676  1.079   1.314
    verificación vs DP CPU   25/25   25/25  25/25  25/25  25/25   25/25

**Lectura, con su límite explícito:** las seis corren la misma fuente, verifican
idéntico, y el rango entero es 0.32-1.63 — ~5x entre la más lenta y la más rápida de
seis arquitecturas de dos vendors. Ninguna se despeña. Eso sostiene "una fuente, seis
GPUs, rendimiento del mismo orden".

Pero **son GPUs distintas**, así que la tabla mide silicio, no el costo de la capa de
portabilidad. Aislar la capa exigiría la misma GPU bajo ambos toolchains — imposible
en este sitio (la flota NVIDIA no tiene HIP nativo, la AMD no tiene CUDA). Decir "la
capa cuesta X%" con estos datos sería inventarlo, y no se hace.

### 1b. Comparación directa con Accelign (mismo hardware, por fin)

Accelign reporta **9-16 TCUPS en RTX PRO 6000** (BMC Bioinformatics 2026). Nosotros
medimos el PRO 6000 en **1.21-1.63 TCUPS**. Es la primera comparación **en el mismo
hardware** de todo el proyecto, y no necesita la salvedad de "otra GPU":

    Accelign (PRO 6000)  .......... 9-16 TCUPS
    genoaligner (PRO 6000) ........ 1.21-1.63 TCUPS
    ------------------------------------------------
    factor ........................ ~6-13x

El orden de magnitud se mantiene, ahora medido contra la misma tarjeta. Es esperado:
Accelign es un kernel optimizado años; el nuestro es correcto y portable, y **la
optimización no se ha intentado**. Esta es la línea base contra la que medir cualquier
trabajo de optimización futuro — y por primera vez, es una línea base válida.

MMseqs2-GPU (~102 TCUPS en 8x L40S) sigue sin ser comparable directamente: es un
banco de 8 GPUs, no una GPU.






## 2. El reparto de fases cambia la conclusión

Tras atribuir correctamente el costo de contexto (§4), el tiempo por ejecución se
reparte así (régimen len=1024, smax=256, 90%):

    warm-up de contexto (1x)   383 ms   (una vez por proceso; FUERA de e2e)
    generar casos (host)        11.8 ms  48.1%
    empaquetar                   1.5 ms   6.3%
    hipMalloc + memset           0.07 ms  0.3%
    memcpy H2D                   6.0 ms  24.5%
    KERNEL                       5.0 ms  20.5%
    memcpy D2H                   0.05 ms  0.2%
    ----------------------------------------
    end-to-end                  24.4 ms

**Lectura:** el kernel es ~20% del tiempo real. El costo dominante es la
generación de casos en host (48%), que es un artefacto del banco de pruebas, no
del producto. Las transferencias H2D (24.5%) sí son costo real de un usuario.
Cualquier titular que reporte solo `kernel-only` describe un quinto del sistema.

## 3. El régimen que NO cuenta: smax por debajo de la distancia real

`len=1024, smax=64, ident=90%` abandonó 1945 de 2000 pares (97%): la distancia
real (~100 sustituciones) excede `smax=64`, así que el kernel retorna -1 SIN
recorrer wavefronts. Su "0.088 TCUPS" está calculado sobre 2.8% de las celdas y no
es una medición de throughput — es la medición del guard.

Esto se detectó y se corrigió en el tool (job 29210654 → 29210656): la sonda ahora
imprime `resolved/abandoned` y calcula el TCUPS sobre las celdas de los pares
RESUELTOS, con el conteo de abandonados al lado. La misma disciplina de
denominador que la fase de paridad necesitó.

**Consecuencia de diseño:** `smax` es un parámetro de régimen, no una constante. El
kernel lo mapea con `block >= 2*smax+1` y el bloque máximo es 1024, así que este
diseño está acotado a **smax <= 511**. Un régimen con distancia mayor necesita otro
mapeo (multi-bloque o tile por diagonal), no un `smax` más grande.

## 4. El "memset de 475 ms" que no era un memset

Las primeras corridas reportaron 94-99% del end-to-end en `hipMalloc + memset`,
con el memset en 460-480 ms tanto para 8 KB como para 8 MB. **Un costo que no
sigue el tamaño de la operación no es la operación.**

Hipótesis: es el bring-up del contexto HIP, que cae sobre la primera llamada a
device — que resultó ser el memset.

**Prueba falsable (job 29210656 → 29210657):** una llamada desechable a device
ANTES de la sección cronometrada. Resultado:

    antes:  hipMalloc + memset = 460-480 ms    warm-up = (no medido)
    después: hipMalloc + memset = 0.06-0.07 ms  warm-up = 376-383 ms

El costo se movió exactamente donde se predijo. Atribución confirmada, no supuesta.
El warm-up se reporta aparte y se excluye de end-to-end (es una vez por proceso).

**Lección de método, la misma de toda la fase:** los dos bugs de esta sesión
(§3, §4) son números que *parecían* resultados sobre mediciones que no medían lo
que decían. El remedio fue hacer que la herramienta se niegue a reportarlos.

## 5. El lanzamiento tiene que casar con el kernel

Job 29210654 dio "136 TCUPS" con el kernel devolviendo -1 para TODOS los pares: el
lanzamiento pasó **0 bytes de shared memory**. El kernel de score declara
`extern __shared__ int smem[]` y mapea diagonal `k = threadIdx.x - s`, así que
necesita a la vez `block >= 2*smax+1` y una asignación real de smem. Con cero, cada
lectura cae en memoria no asignada.

La configuración correcta está en el harness de paridad
(`tests/parity/wfa_parity.cpp`, ~línea 429) — el sitio de lanzamiento que sí fue
validado. **Regla:** la herramienta de banco debe copiar la configuración de
lanzamiento del sitio validado, no reconstruirla.

## 8. PRO 6000: el negativo que era falso (y cómo se corrigió)

Esta sección existía para documentar por qué la PRO 6000 no era construible. **Era
falso, y la corrección vale más que el error.**

**Lo que afirmé:** "no hay CUDA 12.8+ en el sitio; PRO 6000 no es construible con
este toolchain." Con dos datos reales de respaldo:

    CUDA 12.4 --list-gpu-arch  -> termina en compute_90   (sm_120 no existe)
    CUDA 13.0 + capa ROCm 6.4.3 -> 12+ errores en nvidia_hip_runtime_api.h
                                   (cudaMemLocation, cudaDeviceProp::clockRate, ...)

**Lo que hice mal:** busqué CUDA en `/kuhpc/sw/cuda-toolkit/` (solo 12.4) y en los
dos directorios nvhpc *por defecto* (`2024`→12.4, `2025`→13.0). **No recorrí las
instalaciones nvhpc versionadas.** Una de ellas, `25.3`, tiene CUDA **12.8**:

    /kuhpc/sw/nvhpc/Linux_x86_64/25.3/cuda/12.8/bin/nvcc
      --list-gpu-arch  -> ... compute_89, compute_90, compute_120   (¡sm_120!)
      + capa ROCm 6.4.3 -> compila limpio, RC=0

12.8 es la única versión que satisface **ambas** restricciones a la vez: conoce
sm_120, y sigue siendo 12.x así que la capa `nvidia_detail` compila. Exactamente la
condición que §8 decía que no existía. Recorrer las versiones fue un comando.

**Lección, la tercera vez en este proyecto:** un negativo deducido de una búsqueda
parcial es una hipótesis, no un hecho — y aquí lo escribí en el doc como hecho, con
la frase "verificado". El skill ya lo dice (*"when a skill says X is impossible,
re-test X"*); lo que faltaba era aplicarlo a mi propia conclusión del párrafo
anterior. Ahora: **recorrer cada instalación versionada, no solo los paths con
nombre.**

Resultado del job 29210709: medido y verificado 25/25 por régimen. Ver §1 y §1b — y esta
es además la primera comparación directa contra Accelign **en el mismo hardware**.

## 9. Manifiesto de replicabilidad

    repo      alrobles/genoaligner-devel (privado)
    commit    6d27737  (código medido: bench + jobs)
    jobs      29210660 (MI210), 29210661 (RTX 6000), 29210662 (A100),
              29210664 (V100), 29210665 (L40), 29210709 (PRO 6000)
    comando   sbatch scripts/h6_bench.sbatch          # MI210, hipcc
              sbatch scripts/h6_bench_cuda.sbatch     # RTX 6000, nvcc 12.4 sm_75
              sbatch scripts/h6_bench_a100.sbatch     # A100,     nvcc 12.4 sm_80
              sbatch scripts/h6_bench_v100.sbatch     # V100,     nvcc 12.4 sm_70
              sbatch scripts/h6_bench_l40.sbatch      # L40,      nvcc 12.4 sm_89
              sbatch scripts/h6_bench_pro6000.sbatch  # PRO 6000, nvcc 12.8 sm_120

    herramienta   bench/tcus.cpp
    build ROCm    /kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. \
                  -DGENOALIGNER_BENCH_BACKEND='"rocm"' -o tcus bench/tcus.cpp
    build CUDA    nvcc -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
                  -gencode arch=compute_XX,code=sm_XX \
                  -I <rocm>/include -I <cuda>/include \
                  -I <cuda>/targets/x86_64-linux/include -I . -lcuda \
                  -x cu -o tcus_cuda bench/tcus.cpp
    output        /beegfs/a474r867/genoaligner/bench/h6_<dev>_<jobid>.txt

    toolchain     ROCm 6.4.3, hipcc 6.4.43484, AMD clang 19
    device 1      AMD Instinct MI210, gfx90a:sramecc+:xnack-
    toolchain     nvcc 12.4 (nvhpc 2024) para sm_70..sm_89
    device 2      Quadro RTX 6000,  sm_75   (Turing)
    device 3      NVIDIA A100-PCIE-40GB, sm_80   (Ampere)
    device 4      Tesla V100S-PCIE-32GB, sm_70   (Volta)
    device 5      NVIDIA L40,        sm_89   (Ada)
    toolchain     nvcc 12.8 (/kuhpc/sw/nvhpc/Linux_x86_64/25.3/cuda/12.8) para sm_120
    device 6      NVIDIA RTX PRO 6000 Blackwell Server Edition, sm_120  (Blackwell)

Los jobs reconstruyen el binario desde el fuente versionado antes de correr, así que
el artefacto medido es el commit, no un binario suelto. Los números se escriben a
`/beegfs/.../bench/`, durables y con job id en el nombre. Los seis sbatch comparten
los mismos cuatro regímenes byte a byte (mismo orden, mismos flags), para que la
comparación no sea de erratas.

## 10. Pendiente (no medido — no reportar como hecho)

- **Optimización.** 0.32-1.63 TCUPS es un kernel correcto y portable, no uno rápido.
  Las palancas (extensión vectorizada, layout de memoria, occupancy) **no se han
  tocado**. Con §1b hay por fin una línea base válida contra la cual medir el salto:
  Accelign en la misma GPU, 9-16 TCUPS.
- **Regímenes más allá de smax=511** requieren otro mapeo de bloque.
- **`len` > 4096 / ident < 70%** (distancia alta) no se midieron: fuera del régimen
  filogenético que el proyecto apunta, pero es un hueco declarado.
- **V100 en modo no-S (sm_70 puro)**: el nodo reportó un V100S (sm_70 compilado corre,
  pero es la variante S). No cambia la conclusión, se anota por completitud.

