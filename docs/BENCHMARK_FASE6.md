# Fase 6 — Benchmark honesto

> **Medido:** 2026-09-11 · **Backends:** ROCm 6.4.3/hipcc, CUDA 12.4/nvcc
> **GPUs:** MI210, RTX 6000, A100, V100, L40. PRO 6000 no construible (§8)
> **Jobs:** 29210654 (inválido), 29210660/61 (MI210/q6000), 29210662 (A100),
>            29210664 (V100), 29210665 (L40)
> **Herramienta:** `bench/tcus.cpp` · `scripts/h6_bench{,_cuda,_a100,_v100,_l40}.sbatch`
> **HEAD medido:** ea2f9ac

## 0. Qué se afirma y qué no

**Se afirma:** el kernel de score WFA corre desde una sola fuente en **cinco GPUs de
dos vendors** (MI210, RTX 6000, A100, V100, L40), y rinde **0.32-1.48 TCUPS** en los
regímenes donde el trabajo se resuelve completo, verificado contra el DP de CPU sobre
una muestra de pares en cada corrida y en cada GPU.

**No se afirma:** que genoaligner sea competitivo con Accelign (9-16 TCUPS en RTX
PRO 6000) ni con MMseqs2-GPU (~102 TCUPS en 8x L40S). Seguimos ~1 orden por debajo.
El objetivo del proyecto es portabilidad, no récord de TCUPS. Tampoco se afirma que
la capa de portabilidad sea gratis: la matriz mide silicio, no la capa (§1a).

## 1. Resultados — matriz de 5 GPUs (jobs 29210660/61/62/64/65)

`kernel-only` = celdas de los pares RESUELTOS / tiempo del kernel. 1 TCUPS = 1e12
celdas/s. `end-to-end` excluye el warm-up de contexto y la generación de casos (§2, §4).

    kernel-only (TCUPS)
    régimen (resueltos)          MI210/hipcc  RTX6000/nvcc  A100/nvcc  V100/nvcc  L40/nvcc
    len=256,  smax=64,  90%        0.387        0.339       0.446     0.539     1.477
    len=1024, smax=256, 90%        0.419        0.316       0.705     0.530     1.014
    len=1024, smax=256, 70%        0.469        0.431       0.725     0.676     1.079
    len=1024, smax=64,  90%       (inválido: 1945/2000 pares abandonados — ver §3)

    end-to-end (TCUPS)
    régimen                      MI210   RTX6000   A100    V100    L40
    len=256,  smax=64,  90%      0.035    0.105    0.107   0.115   0.213
    len=1024, smax=256, 90%      0.164    0.208    0.333   0.282   0.499
    len=1024, smax=256, 70%      0.165     —       0.333   0.320   0.528

Verificación: **25/25 pares contra el DP de CPU en cada régimen resuelto, en las
cinco GPUs** (3/3 en el régimen abandonado, que es todo lo que hay para comparar).

**Lectura de la matriz:**

    L40  (Ada, sm_89) ...... 1.01-1.48 TCUPS   <- la más rápida, por 1.4-4x
    A100 (Ampere, sm_80) ... 0.45-0.73
    RTX 6000 (Turing, sm_75) 0.32-0.43
    MI210 (AMD, gfx90a) .... 0.39-0.47
    V100 (Volta, sm_70) .... 0.53-0.68         <- el "suelo" NO es el más lento

El V100 supera al RTX 6000 y al MI210 pese a ser la arquitectura más vieja: tiene
mucho más ancho de banda (HBM2, ~900 GB/s vs GDDR6 del q6000). El orden sigue el
ancho de banda de memoria, no la edad ni el vendor — que es lo esperado en un kernel
que es esencialmente una caminata por memoria. El L40 (GDDR6 ~864 GB/s pero
arquitectura Ada y cachés mucho mayores) duplica al A100, así que el ancho de banda
solo no lo explica todo: la arquitectura también pesa. No se afirma una causa única.

### 1a. Los backends entre sí (H6: ¿la abstracción cuesta rendimiento?)

El MISMO `bench/tcus.cpp`, cinco GPUs, dos toolchains:

    régimen                 MI210/hipcc  RTX6000/nvcc  A100/nvcc  V100/nvcc  L40/nvcc
    len=256,  smax=64, 90%     0.387        0.339        0.446     0.539      1.477
    len=1024, smax=256,90%     0.419        0.316        0.705     0.530      1.014
    len=1024, smax=256,70%     0.469        0.431        0.725     0.676      1.079
    verificación vs DP CPU     25/25        25/25        25/25     25/25      25/25

**Lectura, con su límite explícito:** las cinco corren la misma fuente, verifican
idéntico, y ninguna se despeña — el rango entero es 0.32-1.48, menos de 5x entre la
más lenta y la más rápida de cinco arquitecturas distintas de dos vendors. Eso
sostiene "una fuente, cinco GPUs, rendimiento del mismo orden".

Pero **estas son GPUs distintas**, así que la tabla mide silicio, no el costo de la
capa de portabilidad. Aislar la capa exigiría la misma GPU bajo ambos toolchains —
imposible en este sitio (la flota NVIDIA no tiene HIP nativo y la AMD no tiene CUDA).
Decir "la capa cuesta X%" con estos datos sería inventarlo, y no se hace.

Comparación con la literatura, en el hardware donde SÍ corren (no medido aquí):

    Accelign ......... 9-16 TCUPS en RTX PRO 6000     -> ~6-16x sobre nuestro L40
    MMseqs2-GPU ..... ~102 TCUPS en 8x L40S           -> ~250x por encima

La brecha con Accelign es menor de lo que sugerían las primeras dos GPUs, pero sigue
siendo un orden de magnitud. Esperado: 0.4-1.5 TCUPS es un kernel correcto y
portable, no uno optimizado — y la optimización es trabajo que NO se ha intentado.





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

## 8. PRO 6000 (sm_120): por qué NO hay número

El masterplan lista la PRO 6000. **No se puede medir con este toolchain, y la razón
es de versión de CUDA, no de código.** Dos intentos, ambos con evidencia:

    CUDA 12.4 (el que usa todo el repo)   NO soporta sm_120
        nvcc 12.4 --list-gpu-arch -> ...compute_89, compute_90   (máximo sm_90)

    CUDA 13.0 (el que sí soporta sm_120)  ROMPE la capa nvidia_detail de ROCm 6.4.3
        error: no suitable constructor to convert from "int" to "cudaMemLocation"
        error: class "cudaDeviceProp" has no member "clockRate"
        error: class "cudaDeviceProp" has no member "computeMode"
        ... (12+ errores en nvidia_hip_runtime_api.h)

El nvcc 13.0 del sitio sí tiene `compute_120` en su lista de arquitecturas, pero la
capa que traduce HIP→CUDA es de la era 12.x y no compila contra los headers de 13.0.
Es exactamente la restricción ya documentada (CUDA 12.4, no 13.0) mordiendo en la
dirección opuesta: 12.4 no llega a la arquitectura, 13.0 no llega a la capa.

**No hay CUDA 12.8+ en el sitio** (verificado: solo 12.4 en `/kuhpc/sw/cuda-toolkit/`
y en nvhpc 2024/2025). Una 12.8/12.9 cerraría ambas condiciones a la vez y es la
ruta a probar si el sitio la instala.

**Lo que esto NO significa:** que el código no corra en sm_120. Significa que no se
puede *construir* para sm_120 con lo que hay aquí. La conclusión es sobre el
toolchain del sitio, no sobre el kernel.

## 9. Manifiesto de replicabilidad

    repo      alrobles/genoaligner-devel (privado)
    commit    ea2f9ac  (código medido: bench + jobs)
    jobs      29210660 (MI210), 29210661 (RTX 6000), 29210662 (A100),
              29210664 (V100), 29210665 (L40)
    comando   sbatch scripts/h6_bench.sbatch        # MI210, hipcc
              sbatch scripts/h6_bench_cuda.sbatch   # RTX 6000, nvcc sm_75
              sbatch scripts/h6_bench_a100.sbatch   # A100, nvcc sm_80
              sbatch scripts/h6_bench_v100.sbatch   # V100, nvcc sm_70
              sbatch scripts/h6_bench_l40.sbatch    # L40,  nvcc sm_89

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
    toolchain     nvcc 12.4 (nvhpc 2024)
    device 2      Quadro RTX 6000,  sm_75   (Turing)
    device 3      NVIDIA A100-PCIE-40GB, sm_80   (Ampere)
    device 4      Tesla V100S-PCIE-32GB, sm_70   (Volta)
    device 5      NVIDIA L40,        sm_89   (Ada)

Los jobs reconstruyen el binario desde el fuente versionado antes de correr, así que
el artefacto medido es el commit, no un binario suelto. Los números se escriben a
`/beegfs/.../bench/`, durables y con job id en el nombre. Los cinco sbatch comparten
los mismos cuatro regímenes byte a byte (mismo orden, mismos flags), para que la
comparación no sea de erratas.

## 10. Pendiente (no medido — no reportar como hecho)

- **PRO 6000 (sm_120).** No construible con el toolchain del sitio; ver §8.
- **Optimización.** 0.32-1.48 TCUPS es un kernel correcto y portable, no uno rápido.
  Las palancas (extensión vectorizada, layout de memoria, occupancy) **no se han
  tocado** y es el mayor salto disponible. El kernel es esencialmente una caminata
  por memoria y el throughput sigue el ancho de banda, así que ahí está el trabajo.
- **Regímenes más allá de smax=511** requieren otro mapeo de bloque.
- **`len` > 4096 / ident < 70%** (distancia alta) no se midieron: fuera del régimen
  filogenético que el proyecto apunta, pero es un hueco declarado.
- **Comparación directa con Accelign/WFA-GPU en el mismo hardware.** No corren en
  MI210, así que la comparación es contra números publicados en otra GPU.

