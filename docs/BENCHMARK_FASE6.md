# Fase 6 — Benchmark honesto

> **Medido:** 2026-09-11 · **Backend:** ROCm 6.4.3/hipcc y CUDA 12.4/nvcc
> **GPUs:** AMD Instinct MI210, Quadro RTX 6000
> **Jobs:** 29210654 (inválido), 29210656/57, 29210658, 29210660/61 (finales)
> **Herramienta:** `bench/tcus.cpp` · **Jobs:** `scripts/h6_bench{,_cuda}.sbatch`
> **HEAD medido:** 69a1603

## 0. Qué se afirma y qué no

**Se afirma:** el kernel de score WFA corre en el MI210 y en el RTX 6000 desde la
misma fuente, y rinde **0.32-0.47 TCUPS** en los regímenes donde el trabajo se
resuelve completo (pares de 256-1024 bp con la distancia real por debajo de
`smax`), verificado contra el DP de CPU sobre una muestra de pares en cada corrida.

**No se afirma:** que genoaligner sea competitivo con Accelign (9-16 TCUPS en RTX
PRO 6000) ni con MMseqs2-GPU (~102 TCUPS en 8x L40S). Estamos 1-2 órdenes por
debajo. El objetivo del proyecto es portabilidad, no récord de TCUPS.

## 1. Resultados (jobs 29210660 MI210, 29210661 RTX 6000)

`kernel-only` = celdas de los pares RESUELTOS / tiempo del kernel.
`end-to-end` = lo que ve un usuario: empaquetado + alloc + transferencias + kernel,
**sin** el warm-up de contexto ni la generación de casos (ver §2 y §4).
Ambos usan 1 TCUPS = 1e12 celdas/s.

    régimen                          resol.   kernel-only        end-to-end
    len=256,  smax=64,  8000 pares    8000/8000   0.387 TCUPS     0.035 TCUPS
    len=1024, smax=256, 2000 pares    2000/2000   0.419 TCUPS     0.164 TCUPS
    len=1024, smax=256, 2000 pares    2000/2000   0.469 TCUPS     0.165 TCUPS   (ident=70%)
    len=1024, smax=64,  2000 pares      55/2000     (inválido, ver §3)

Verificación: 25/25 pares contra el DP de CPU en cada régimen resuelto (3/3 en el
régimen abandonado, que es todo lo que hay para comparar).

### 1a. Los dos backends (H6 del masterplan: ¿la abstracción cuesta rendimiento?)

El MISMO `bench/tcus.cpp` bajo hipcc y nvcc, mismos regímenes:

    régimen                    MI210/hipcc   RTX 6000/nvcc   diff
    len=256,  smax=64,  90%      0.387          0.339        -12%
    len=1024, smax=256, 90%      0.419          0.316        -25%
    len=1024, smax=256, 70%      0.469          0.431         -8%
    verificación vs DP de CPU    25/25          25/25

**Lectura:** dentro del ~25%, que es diferencia de silicio y compilador, no un
colapso de la capa de portabilidad — la fuente es la misma y ninguno de los dos
lados se despeña. **Advertencia:** son GPUs distintas, así que esto NO aísla el
costo de la capa. Aislarlo exigiría el mismo silicio bajo ambos toolchains, que no
está disponible aquí. Decir "la capa cuesta X%" con estos datos sería inventarlo.

Comparación con la literatura, en el hardware donde SÍ corren (no medido aquí):

    Accelign ......... 9-16 TCUPS en RTX PRO 6000     -> ~20-40x por encima
    MMseqs2-GPU ..... ~102 TCUPS en 8x L40S           -> ~250x por encima

Esperado: 0.4 TCUPS es un kernel correcto y portable, no uno optimizado.



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

## 6. Manifiesto de replicabilidad

    repo      alrobles/genoaligner-devel (privado)
    commit    1c8b71a  (código medido; HEAD al lanzar los jobs)
              los docs de este archivo se commitearon en 2869821
    jobs      29210657 (MI210), 29210658 (RTX 6000)
    comando   sbatch scripts/h6_bench.sbatch        # MI210, hipcc
              sbatch scripts/h6_bench_cuda.sbatch   # RTX 6000, nvcc

    herramienta   bench/tcus.cpp
    build ROCm    /kuhpc/sw/rocm/6.4.3/bin/hipcc -O2 -std=c++17 -I. \
                  -DGENOALIGNER_BENCH_BACKEND='"rocm"' -o tcus bench/tcus.cpp
    build CUDA    nvcc -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
                  -gencode arch=compute_75,code=sm_75 \
                  -I <rocm>/include -I <cuda>/include \
                  -I <cuda>/targets/x86_64-linux/include -I . -lcuda \
                  -x cu -o tcus_cuda bench/tcus.cpp
    output        /beegfs/a474r867/genoaligner/bench/h6_{rocm,cuda}_<jobid>.txt

    toolchain     ROCm 6.4.3, hipcc 6.4.43484, AMD clang 19
    device 1      AMD Instinct MI210, gfx90a:sramecc+:xnack-
    toolchain     nvcc 12.4.131 (nvhpc 2024)
    device 2      Quadro RTX 6000, sm_75

Ambos jobs reconstruyen el binario desde el fuente versionado antes de correr, así
que el artefacto medido es el commit, no un binario suelto. Los números se escriben
a `/beegfs/.../bench/`, durables y con job id en el nombre.

## 7. Pendiente (no medido — no reportar como hecho)

- **A100 y PRO 6000.** El masterplan los lista; solo se midieron MI210 y RTX 6000.
  Ambos nodos existen en el cluster (verificado con `scontrol`), el job solo cambia
  el `--gres` y el `-gencode` (a100 = sm_80, pro6000 = sm_120).
- **Generación de casos fuera del cronómetro.** Hoy ~50% del end-to-end es generar
  los pares en host; el banco debe pregenerarlos para medir lo que al usuario le
  importa. Es un artefacto del banco, no del producto.
- **`smax` > 511** no es alcanzable con este mapeo de bloque.
- **Optimización.** 0.4 TCUPS es un kernel correcto y portable, no uno rápido. Las
  palancas (extensión vectorizada, layout, occupancy) no se han tocado.
- **Comparación directa con Accelign/WFA-GPU en el mismo hardware.** No corren en
  MI210, así que la comparación es contra números publicados en otra GPU.

