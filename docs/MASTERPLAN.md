# MASTERPLAN — genoaligner: primer producto de software

> **Repo:** `alrobles/genoaligner-devel` (privado)
> **Autor:** Ángel Luis Robles Fernández — Reuman Lab / KU
> **Fecha:** 2026-09-10 · **Versión:** 1.0
> **Objetivo estratégico:** destrabar la inferencia filogenética de phylogenyAI
> con infraestructura de alineación propia, portable CUDA+ROCm.

---

## 0. Tesis: por qué esto, por qué ahora

El proyecto no nace de un capricho técnico. Nace de tres activos que ya
tenemos y que la competencia no puede replicar barato:

### 0.1 Sandbox casi infinito de hardware heterogéneo

    NVIDIA (CUDA):  q6000 ×~29, A100 ×~18, L40 ×4, pro6000 ×5, V100, A40, q8000
    AMD (ROCm):     MI210 ×~81  (27 nodos × 3)
    Toolchains:     cuda/13.0 + rocm/6.4.3, ambos verificados funcionando

Nadie en filogenética está usando las MI210. **Hardware ocioso = ventaja
gratuita.**

### 0.2 Diseño para el fallo (test-and-drop barato)

De `ecoreasoner/ROADMAP.md` §7, reglas aprendidas con sangre:

> **test-and-drop: hipótesis validable en <4h GPU ANTES de escalar.**
> Pool teórico ≠ pool real — medir con `scontrol`, no `sinfo`.

Esta es la filosofía que gobierna el masterplan. No "construir el alineador
perfecto", sino **ciclos baratos de hipótesis → medir → descartar o escalar**,
miles de veces, con agentes + cluster.

### 0.3 Software factory agéntica operativa

    hermes-hydra        router/balanceador; exec en ku-hpc, reumanlab, alpha, terminal
    knowledgebase       patrón HPC canónico (ARCHITECTURE.md), tier system T0/T1/T2
    ecoreasoner         harness Slurm con checkpoint/resubmit (el patrón a copiar)
    devin CLI           asistente de código, gratis por promoción sin tope
    Hermes (local)      planificador + ejecutor verificado

**El mejor activo no es el código: es la capacidad de generar y descartar
diseños rápido.** El masterplan está construido para explotar eso.

---

## 1. Objetivo del primer producto

**Un pipeline de alineación portable que corra en MI210 y produzca resultados
idénticos a una referencia CPU — con WFA como primer método y SW como segundo.**

Criterio de éxito explícito y binario:

    genoaligner corre en MI210 (gfx90a ROCm)
    Y produce scores idénticos a SeqAn/Parasail sobre un dataset de control
    Y es replicable en otro cluster desde un manifiesto

NO es criterio: "más rápido que Accelign en TCUPS".

### Por qué WFA primero, SW después (decisión del usuario)

- **WFA primero:** es el método apropiado para filogenética (homólogos,
  distancia de edición baja → coste bajo). Es el kernel más simple de
  validar y el que destraba phylogenyAI.
- **SW después:** no lo necesitamos, pero hay gente que sí → visibilidad y
  adopción. Estrategia de producto, no de necesidad técnica.

---

## 2. Arquitectura propuesta (a validar con test-and-drop)

### 2.1 Decisión central: HIP puro ✅ VALIDADA (con una corrección)

    Una base de código HIP → hipcc  → ROCm nativo (MI210)
                          → nvcc con -D__HIP_PLATFORM_NVIDIA__ → CUDA

Razones: es lo que AMD usa (minimap2 heterogéneo, jul 2026); el cluster lo
tiene; evita la deuda técnica de hipify; SYCL no está listo en el cluster
(falta libs GPU de oneAPI); Kokkos añade una capa innecesaria para 2 targets.

**Test-and-drop:** validar HIP puro con un kernel trivial en ambas
plataformas ANTES de escribir el kernel real. Si falla, retroceder a SYCL.

**VEREDICTO (2026-09-11): la decisión es correcta, pero esta sección la
explicaba mal.** El renglón `hipcc con -D__HIP_PLATFORM_NVIDIA__` es engañoso:
no es hipcc el que compila para NVIDIA (hipcc es un wrapper de clang de AMD y
NO envuelve a nvcc). El compilador para NVIDIA es **`nvcc`**, y lo que hace
posible la fuente única es una **capa de headers** dentro de ROCm:

    hip/hip_runtime.h
      #if defined(__HIP_PLATFORM_NVIDIA__) && !defined(__HIP_PLATFORM_AMD__)
        #include <hip/nvidia_detail/nvidia_hip_runtime.h>   // -> <cuda_runtime.h>
      #elif ...

Así que "una fuente, dos backends" se sostiene — pero por include paths y un
`-D`, no porque hipcc sepa hablar con CUDA. Detalles y la invocación exacta en
la Fase 1; el requisito medido es **CUDA 12.4, no 13.0**.

Corolario de método: durante H1 se buscó un *shim instalable*, no se encontró, y
se concluyó que la tesis estaba muerta para NVIDIA. La conclusión era falsa: el
mecanismo no era un paquete, era un header ya presente. Un negativo
"documentado" sigue siendo una hipótesis hasta que se reproduce en el caso
concreto.

### 2.2 Estructura del repo

```
genoaligner-devel/
├── README.md
├── CMakeLists.txt              # build dual: detecta ROCm o CUDA
├── docs/
│   ├── OPTION_A_ANALYSIS.md
│   └── MASTERPLAN.md           # este documento
├── include/genoaligner/
│   ├── types.hpp               # CIGAR, scoring scheme, resultado
│   ├── api.hpp                 # interfaz pública (backend-agnostic)
│   └── backend/
│       ├── hip_common.hpp      # abstracción HIP (única para ambos)
│       └── dispatch.hpp        # RT/device selection
├── src/
│   ├── wfa/
│   │   ├── wfa_kernel.hip      # kernel WFA (el corazón, ~600-800 líneas)
│   │   ├── wfa_host.cpp        # lanzamiento, buffers, etapa
│   │   └── wfa_traceback.hip   # (fase posterior)
│   ├── sw/
│   │   └── sw_kernel.hip       # Smith-Waterman (segundo método)
│   ├── io/fasta.cpp            # lectura/escritura
│   └── reference/seqan_ref.cpp # referencia CPU para validación
├── tests/
│   ├── unit/                   # tests por componente
│   ├── parity/                 # paridad vs SeqAn (el test que importa)
│   └── data/                   # datasets de control (pequeños, versionados)
├── bench/
│   ├── tcus.cpp                # medición de TCUPS
│   └── slurm/                  # jobs de benchmark en el cluster
└── scripts/
    ├── build_rocm.sh
    ├── build_cuda.sh
    └── validate_parity.py
```

### 2.3 El kernel WFA — qué hay que implementar

WFA (Wavefront Alignment) computa el alineamiento en O(ns) donde s = distancia
de edición. Para filogenética (identidad 50-95%) s es pequeño → muy eficiente.

Componentes del kernel (basado en la formulación publicada, no en copia):

1. **Wavefront storage:** por cada diagonal k y cada score s, guardar la
   extensión máxima. Mapeo a memoria GPU (compartida para wavefronts activos).
2. **Extensión:** cada hilo extiende un match desde un offset dado.
3. **Reducción:** combinar los tres predecesores (I, D, M) por celda.
4. **Bucle de score** hasta alcanzar el criterio de parada.
5. **Offset (WFA-adaptive):** manejar wavefronts lejos de la diagonal.

Los puntos difíciles (donde test-and-drop paga): asignación de memoria para
wavefronts variables, sincronización entre iteraciones de score, y traceback.

---

## 3. Fases — diseñadas para fallo barato

Cada fase tiene un **criterio de fusión** (advance/kill) medible en <4h GPU.
Si no se cumple, se descarta el enfoque y se prueban alternativas. **Ningún
diseño se compromete sin pasar su test.**

### Fase 0 — Verificación de plataforma ✅ COMPLETA

Entregable: informe de toolchain.
Estado: HIP corrió en MI210; CUDA 13.0 presente. Ver `OPTION_A_ANALYSIS.md`.

### Fase 1 — Esqueleto dual + kernel trivial (1 semana) ✅ COMPLETA

- CMake que detecte ROCm/CUDA y compile la misma fuente HIP.
- Kernel trivial (`vector_add`) que corra en MI210 y en una NVIDIA.
- Script de build para ambos + CI mínima (compila en los dos).

**Criterio de fusión:** el mismo binario fuente compila y corre correcto en
MI210 (ROCm) Y en q6000 (CUDA). Si no → evaluar SYCL.

**RESULTADO (2026-09-11): CUMPLIDO.** No con un `vector_add`, sino con el kernel
WFA real, que es una prueba mucho más fuerte. La misma fuente `wfa_kernel.hip`
compila y corre en ambos:

    MI210  (gfx90a, hipcc 6.4.3) ....... 100.00%  (job 29184155)
    RTX 6000 (Turing sm_75, nvcc 12.4) .. 100.00%  (job 29199289)

Mismo set de control, mismos números (946 pass / 0 fail / 61 abandoned).

**CÓMO, y por qué el §10 original se equivocaba.** El masterplan y buena parte
del trabajo de H1 asumieron que la portabilidad HIP-pura a NVIDIA requería un
*shim instalable*, y al no encontrarlo se concluyó (demasiado rápido) que la
tesis estaba muerta para NVIDIA. No lo está. El mecanismo siempre estuvo en los
headers de ROCm:

    hip/hip_runtime.h  --(si __HIP_PLATFORM_NVIDIA__)-->  hip/nvidia_detail/
                                                          nvidia_hip_runtime.h
                                                          -> #include <cuda_runtime.h>

Es una **capa de headers, no un toolchain aparte**. Compilar para NVIDIA es:

    nvcc -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
         -gencode arch=compute_75,code=sm_75 \
         -I $ROCM/include -I $CUDA/include -I $CUDA/targets/x86_64-linux/include \
         -x cu ...

Sin hipify, sin instalar nada, sin dependencia nueva. La única restricción
medida: **CUDA 12.4, no 13.0** (el layer de ROCm 6.4.3 apunta a 12.x y falla con
13.0 en `cudaMemLocation` / `cudaDeviceProp.clockRate`).

Lo que faltaba no era el mecanismo, era **la prueba**. Diez minutos de `nvcc`
directo desmintieron una conclusión que se había apoyado en un skill en vez de
en una reproducción.

### Fase 2 — Kernel WFA score-only (2 semanas) ✅ COMPLETA

- Implementación WFA sin traceback, solo score.
- Manejo de memoria de wavefronts.

**Criterio de fusión:** score correcto en ≥95% de un set de control de
~1,000 pares (comparado contra referencia CPU). <95% → revisar formulación
antes de optimizar.

**RESULTADO (2026-09-11): CUMPLIDO con 100%.** Job 29184155 en MI210
(`gfx90a:sramecc+:xnack-`, r06r18n01): 946 resueltos, 946 pass, 0 fail,
61 abandonados (`isD > smax=64`), paridad 100.00%.

Coste real: 2 submits + 1 smoke job, no los "2h" estimados. El primer submit
(job 29184154) falló con `Memory access fault` (rc=134) por un buffer de
wavefront un `int` corto — un bug que **el gate CPU no podía ver**, porque la
rama shim del kernel re-dimensionaba el buffer por su cuenta. Detalles y
aritmética en `ROADMAP_RETAKEOVER.md` §8.

Nota de método: las cuatro primeras formulaciones fallaron por deducir el
álgebra en vez de portar la referencia. El port verbatim de
`wavefront_compute_edit_idm` (con cita archivo:línea) resolvió en una iteración
lo que cuatro intentos de deducción no resolvieron.

### Fase 3 — Paridad exacta + QA (1 semana) ✅ COMPLETA

- Set de control: pares con identidad conocida (50%, 70%, 90%, 100%).
- Comparación exhaustiva contra SeqAn/Parasail.
- Reporte de paridad (100% esperado; documentar cualquier divergencia).

**Criterio de fusión:** 100% de scores idénticos a la referencia. Si hay
divergencias → entender por qué ANTES de seguir (puede ser bug o límite).

**RESULTADO (2026-09-11): CUMPLIDO con 100%, en AMBOS backends.** Jobs 29201335
(MI210) y 29202787 (RTX 6000), números idénticos:

    cases      : 1007
    resolved   : 946
    agree      : 946
    mismatch   : 0
    abandoned  : 61   (isD > smax=64; NO es fallo)
    parity     : 100.00%
    seqan3     : 946 filas verificadas, 0 desacuerdos

**La referencia son TRES implementaciones que no escribimos nosotros**, no una:

    edlib       Myers bit-vector, C (binding Python)
    rapidfuzz   Levenshtein bit-parallel, código C++ independiente
    SeqAn3      header-only C++23, 3.4.0

Más el DP propio (`src/reference/edit_distance_cpu.hpp`) como cuarta. Las tres
externas se verifican entre sí en cada par; si dos discrepan, el oráculo falla
en vez de emitir un número. En los 300 pares del set de control:
edlib == rapidfuzz == seqan3, cero desacuerdos.

**Por qué la distinción con H2 importa.** H2 (jobs 29184155, 29199289) comparó
la GPU contra nuestro propio DP. Eso es necesario pero no suficiente para la
afirmación de Fase 3: ambos lados compartirían cualquier error conceptual
nuestro. Contra implementaciones externas, ese riesgo desaparece.

**Corrección a esta sección: Parasail NO se usó, y SeqAn SÍ.** Durante R5
afirmé que SeqAn era "un proyecto C++ aparte" y descarté ambos. Medido después:

    SeqAn3 es header-only. git clone --depth 1 = 4.3 MB, segundos.
    Compila con /kuhpc/sw/gcc/14.2/bin/g++ -std=c++23 (el g++ por defecto del
    cluster es 11.5 y NO sirve: SeqAn3 3.4 exige GCC >= 12 y C++23).
    Coste real: ~25 min, incluidos 5 intentos de compilación por errores de API
    (method_global{} | edit_scheme; assign_char en vez de from_char; las views
    se movieron a utility/views).

    Parasail: instalable por pip, pero su binding de Python está roto para coste
    unitario en 1.3.4 — matrix_create ignora match/mismatch (produce size n+1 y
    diagonal == size), Matrix(name) es inmutable, y Matrix(file) da una matriz
    correcta que nw() luego puntúa como ACGT/ACGT = 1 con todos los gaps
    probados. Nueve iteraciones sin resultado.

Así que la mención del masterplan era correcta en el caso de SeqAn y mi
justificación para descartarla era falsa; y en el caso de Parasail al revés
(instalable pero inservible). **El patrón que se repite: deducir la API en vez
de leer la fuente autoritativa.** Con SeqAn el ejemplo funcional estaba dentro
del propio header, en `align_pairwise.hpp`.

Nota de convención: SeqAn3 maximiza, así que `edit_scheme` puntúa match 0,
mismatch −1, gap −1. La distancia de edición es la **negación** del score.

### Fase 4 — Traceback (2-3 semanas — la fase de riesgo) ✅ COMPLETA

- Reconstrucción del CIGAR. Es donde WFA-GPU dedica más complejidad.
- Estrategia de memoria para almacenar el camino.

**Criterio de fusión:** CIGAR correcto → el alineamiento reconstruido
reproduce el score. Es un test **autoconsistente** (no necesita referencia).

**Resultado (2026-09-11): COMPLETA, y el criterio autoconsistente resultó
insuficiente.** El CIGAR se extrae del kernel en ambos backends con la misma
fuente. Dos bugs de índice (pases forward multi-diagonal; arranque de la caminata
en el wavefront 0 en vez de `score_total`), ambos silenciosos en GPU, ambos
hallados al separar `unresolved` en "distancia real > smax" vs "dentro de smax".

    MI210  (job 29207419) ... H4 PASS · 203/203 re-score · 203/203 bien formados · gate edlib GPU 203/203
    RTX 6000 (job 29207420) ... H4-CUDA PASS · idéntico · gate edlib GPU 203/203

Nota: re-score y buena formación son autoconsistencia — ambos pasaron mientras el
CIGAR estaba mal durante el desarrollo. El oráculo externo (edlib) se añadió como
Stage 3 del gate y corre ahora dentro del job de GPU. La variante shared-memory no
está implementada (el kernel usa workspace global); la corrida 0 B vs 48 KB solo
mide que el resultado no cambia. Ver `ROADMAP_RETAKEOVER.md` §R7.

### Fase 5 — Capa de portabilidad formal (1 semana) ✅ COMPLETA

- Abstracción de backend completa.
- Suite de tests que corre idénticos en ambas plataformas.

**Criterio de fusión:** todos los tests de fases 2-4 pasan en ROCm Y CUDA
desde el mismo fuente.

**RESULTADO (2026-09-11): CUMPLIDO.** Un `CMakeLists.txt`, un árbol de fuentes,
dos compiladores:

    ROCm/MI210  job 29207287 ... backend ROCm, MI210 gfx90a, round-trip PASS
    CUDA/q6000  job 29207294 ... backend CUDA, RTX 6000 sm_75, round-trip PASS

Además, el test previo contra Fases 2-3 (paridad 100% vs 3 oráculos externos) ya
corría en ambos backends: jobs 29201335 (MI210) y 29202787 (RTX 6000).

**Corrección de fondo sobre la arquitectura.** El `CMakeLists` anterior exigía
`CMAKE_CXX_COMPILER=hipcc` para **ambos** backends, lo que hacía la rama CUDA
imposible por construcción: hipcc es el wrapper de clang de AMD y nunca invoca a
nvcc, y no existe shim HIP-para-NVIDIA instalable (cuatro rutas cerradas,
2026-09-10). El modelo correcto es **una fuente, dos compiladores**:

    AMD     hipcc -> ROCm nativo
    NVIDIA  nvcc  -> -D__HIP_PLATFORM_NVIDIA__

El puente es una capa de headers de ROCm, no un compilador:

    hip/hip_runtime.h
      #if defined(__HIP_PLATFORM_NVIDIA__) && !defined(__HIP_PLATFORM_AMD__)
        #include <hip/nvidia_detail/nvidia_hip_runtime.h> -> <cuda_runtime.h>

**Cuatro incompatibilidades específicas de CUDA, encontradas en este orden** —
las tres primeras solo aparecen en un build dirigido por CMake, no en una
invocación manual de nvcc sobre un fichero:

    1. La propiedad LANGUAGE CXX en .hip debe fijarse ANTES de add_executable.
       Hacerlo dentro de la función helper es demasiado tarde:
       "Cannot determine link language for target".
    2. nvcc no reconoce la extensión .hip:
       "nvcc fatal : Don't know what to do with 'probe.hip'" -> hace falta -x cu.
    3. CUDA 12.4 rechaza GCC > 13 como compilador host:
       "unsupported GNU version! gcc versions later than 13 are not supported!"
       El cluster necesita gcc 14.2 para cmake (GLIBCXX_3.4.32) pero nvcc lo
       rechaza -> -ccbin /usr/bin/g++ (gcc 11.5) para el código host.
    4. nvcc no emite objetos position-independent y el enlazador por defecto es
       PIE: "relocation R_X86_64_32S against '.rodata' ... recompile with -fPIE"
       -> -Xcompiler -fPIE en compilación y -Xcompiler -pie en enlace.

Ninguna afecta a ROCm: hipcc trae su propio clang y resuelve las cuatro solo.

Ficheros: `CMakeLists.txt` (reescrito), `scripts/build_cuda.sh` (usa nvcc, ya no
hipcc), `scripts/build_rocm.sh`, `scripts/f5_portability{,_cuda}.sbatch` y
`tests/parity/validate_fase5.sh` (validador sin GPU que además comprueba que la
capa RECHAZA un compilador equivocado).

### Fase 6 — Benchmark honesto (1 semana) ⏳ PARCIAL

- TCUPS en MI210, q6000, A100, pro6000.
- Speedup vs referencia CPU (SeqAn).
- Comparación documentada vs Accelign/WFA-GPU **donde corren** (reconociendo
  que no corren en MI210).

**Criterio de fusión:** números publicables + manifiesto de replicabilidad.

**Resultado (2026-09-11): cinco GPUs de dos vendors medidas y verificadas; PRO 6000
no construible con el toolchain del sitio (ver `docs/BENCHMARK_FASE6.md` §8).**

    kernel-only (TCUPS)      MI210   RTX6000   A100    V100    L40
    len=256,  smax=64,90%    0.387    0.339   0.446   0.539   1.477
    len=1024, smax=256,90%   0.419    0.316   0.705   0.530   1.014
    len=1024, smax=256,70%   0.469    0.431   0.725   0.676   1.079
    verificación vs DP CPU   25/25    25/25   25/25   25/25   25/25

Cinco arquitecturas (Volta, Turing, Ampere, Ada, gfx90a), una fuente, verificación
idéntica, y el rango entero 0.32-1.48 TCUPS: ninguna se despeña. El orden sigue el
ancho de banda de memoria (el V100 adelanta al RTX 6000 y al MI210), no la edad ni el
vendor. Esto mide silicio, no el costo de la capa de portabilidad — aislar eso
exigiría el mismo silicio con ambos toolchains, imposible aquí. Seguimos ~1 orden por
debajo de Accelign (9-16 TCUPS); esperado, el criterio es portabilidad, no récord.

**PRO 6000:** CUDA 12.4 (el del repo) no soporta sm_120; CUDA 13.0 (que sí lo
soporta) rompe la capa `nvidia_detail` de ROCm 6.4.3 con `cudaMemLocation`/`clockRate`.
No hay CUDA 12.8+ en el sitio. Es una limitación del toolchain, no del kernel.

Tres errores de medición corregidos en el proceso, todos del mismo tipo (un número
que parece resultado sobre una medición que no mide lo que dice): un TCUPS calculado
sobre pares abandonados, un "memset de 475 ms" que era el bring-up del contexto HIP, y
la generación de casos en host contada como costo del producto. La herramienta ahora
se niega a reportar los dos primeros (exit codes 5 y 6). `smax` está acotado a 511 por
el mapeo de bloque del kernel.


### Fase 7 — Smith-Waterman (2 semanas, segundo método)

- Kernel SW antidiagonal (mapeo distinto al de WFA).
- Reusa toda la infraestructura de las fases 1-6.

**Criterio de fusión:** los mismos tests de paridad que WFA.

### Fase 8 — Integración en phylogenyAI + paper (2 semanas)

- Enchufar genoaligner al pipeline (donde aporte).
- Manifiesto de despliegue para otro cluster.
- Preprint (JOSS o BMC Bioinformatics).

---

## 4. La filosofía test-and-drop aplicada

Esta es la parte que hace el masterplan distinto de un plan normal. **Cada
hipótesis técnica es un experimento barato, no una decisión de arquitectura.**

### 4.1 Hipótesis a testear (con su coste)

| # | Hipótesis | Test | Coste | Si falla |
|---|---|---|---|---|
| H1 | HIP puro compila a ambos | vector_add dual | 1h | probar SYCL |
| H2 | WFA score-only es correcto | 1,000 pares vs CPU | 2h | revisar formulación |
| H3 | Paridad exacta alcanzable | set de control | 2h | documentar tolerancia |
| H4 | Traceback es viable en GPU | test autoconsistente | días | kernel score-only entregable |
| H5 | Memoria escala a n>1000 | test de estrés | 1h | rediseñar storage |
| H6 | La abstracción no cuesta perf | benchmark ROCm vs CUDA | 2h | aceptar overhead |
| H7 | SW reusa la infraestructura | port del kernel WFA | 1d | infraestructura no general |

**Cada test es <4h (regla de ecoreasoner). Si un test tarda más, la
hipótesis está mal planteada, no el diseño.**

### 4.2 Ciclos agénticos

El patrón de trabajo:

    1. Hermes (local) escribe la hipótesis + el test mínimo
    2. Devin (CLI, gratis) implementa variantes del kernel en paralelo
    3. Hermes lanza el test en el cluster (job Slurm <4h)
    4. Hermes lee el resultado, decide: escalar / descartar / reformular
    5. Repetir

Miles de ciclos posibles porque: el cómputo es ocioso, Devin es gratis, y
cada test es barato por diseño. **El cuello de botella es la calidad de las
hipótesis, no los recursos.**

### 4.3 Diseño para el fallo

Todo está construido asumiendo que partes van a fallar. El plan no apuesta
todo a una carta:

- Si el traceback se atasca → Fase 2-3 ya son entregable + paper.
- Si HIP puro no da paridad → hipify como retroceso.
- Si WFA no escala → SW (Fase 7) ya está planeado.
- Si nada funciona → el informe negativo es un resultado.

---

## 5. Uso de la infraestructura existente (no reinventar)

### 5.1 Lanzamiento de cómputo

**Copiar el patrón de `ecoreasoner`**, no inventar:

- `#SBATCH --signal=B:USR1@300` + `finalize()` idempotente para
  checkpoint/resubmit antes del muro de 6h.
- `AUTO_RESUBMIT=1` para jobs que exceden `sixhour`.
- GPUs por `--gres=gpu:tipo:n`; familias homogéneas por partición.
- **Medir el pool con `scontrol`** (AllocTRES vs Gres), nunca `sinfo`.
- Apptainer para entornos reproducibles si hace falta.

### 5.2 Ejecución remota

Vía `hermes-hydra` (`ku-hpc raw:` para comandos; `/v1/jobs` para largos), o
SSH directo como se ha hecho hoy. Mantener el flujo que ya funciona.

### 5.3 Asistentes de código

Devin CLI (gratis, sin tope) para implementación paralela de variantes de
kernel. Hermes local para planificación, verificación y decisión.

### 5.4 Repo como fuente de verdad

Regla de ecoreasoner §7.5: **sync a repo SIEMPRE** tras tocar scripts en el
cluster. El repo es la verdad, no el estado efímero del HPC.

---

## 6. Lo que destraba esto en phylogenyAI

Conectar el masterplan con el objetivo original:

    HOY:    MAFFT alinea 31 genes en 12 min → la alineación NO es el cuello
            El cuello es el MCMC (semanas) y, al escalar, el volumen

    FUTURO: si el pool crece a 1,000+ genes o genomas completos,
            la alineación pairwise se vuelve el cuello (6.7h → días)
            → ahí genoaligner aporta, y aporta EN LAS MI210

**genoaligner no acelera phylogenyAI hoy.** Lo prepara para cuando el pool
crezca — que es exactamente cuando un competidor que solo tiene CUDA se
quedaría atrás. Es infraestructura para la trayectoria, no para hoy.

---

## 7. Riesgos y mitigaciones

| Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|
| Traceback más difícil de lo estimado | Alta | Medio | Fase 2-3 ya son entregables |
| HIP puro no da paridad exacta | Media | Alto | hipify como retroceso (Fase 1 test lo detecta temprano) |
| Kernel propio 2-5× más lento que Accelign | Alta | Bajo | el criterio es portabilidad, no TCUPS |
| Alcance se expande sin control | Media | Alto | criterios de fusión binarios por fase; techo en Fase 6 |
| Competencia publica port primero | Baja | Medio | el hueco es de nicho; velocidad de iteración manda |
| Presupuesto de Devin cambia | Baja | Bajo | el código queda en el repo; Hermes puede seguir solo |

---

## 8. Decisiones pendientes

1. **¿HIP puro confirmado?** (recomendado; test H1 lo valida en 1h)
2. **¿WFA-adaptive o WFA estándar primero?** (adaptive es más complejo;
   estándar basta para validar el pipeline)
3. **¿Llama a CIGAR propia o SAM-compatible?** (SAM da interoperabilidad)
4. **¿Primer entregable público o todo interno hasta Fase 8?**

---

## 9. Próxima acción inmediata

**Arrancar Fase 1:** esqueleto dual + kernel trivial, test H1.

Concretamente: crear `CMakeLists.txt` dual (ROCm/CUDA), un `vector_add` en
HIP, scripts `build_rocm.sh`/`build_cuda.sh`, y un job Slurm que valide que
el mismo fuente corre en MI210 y en q6000.

Coste: ~1h. Decide la arquitectura de todo lo demás.

---

## 10. Resultado del test H1 (2026-09-10)

Primera ejecución de la Fase 1. Resultado **parcial, y por eso valioso**.

    H1_ROCM (job 29067813): PASS
      backend : ROCm (native HIP)
      device  : AMD Instinct MI210
      arch    : gfx90a:sramecc+:xnack-
      memory  : 63.98 GB
      RESULT  : PASS (round-trip on 1,048,576 elements)

    H1_CUDA (job 29067814): FAIL
      host    : r15r10n01, Quadro RTX 6000, nvcc 13.0
      causa   : hipcc NOT FOUND -- el cluster no tiene el shim HIP de NVIDIA

### Hallazgo

El cluster KU solo provee el toolchain HIP de AMD
(`/kuhpc/sw/rocm/{6.1.0,6.2.1,6.3.1,6.4.1,6.4.3,latest}/bin/hipcc`). No hay
shim de NVIDIA: nvhpc no lo trae, CUDA 13.0 tampoco, y no existe módulo `hip`.

Deducción estratégica: **la flota AMD (~81 MI210) es la única que compila HIP
nativamente sin dependencias externas.** El silicio que ninguna herramienta de
alineación soporta es justamente el que nuestro toolchain domina sin fricción.

### Opciones para la capa NVIDIA

1. **Instalar el shim HIP de NVIDIA** (`hip-nvcc` / toolchain ROCm para
   NVIDIA), en el conda env o compilado. Es el mecanismo oficial: hipcc con
   `-D__HIP_PLATFORM_NVIDIA__` envuelve a nvcc.
2. **hipify + nvcc** para NVIDIA, hipcc directo para AMD. Menos limpio, cero
   dependencias nuevas.

### Coste de descubrir esto ahora

4 iteraciones de ~2 minutos (cmake ausente → cmake ABI → registro .hip →
gres faltante). El test-and-drop funcionó: el límite se encontró antes de
escribir una sola línea del kernel real, no después de meses.

### Consecuencia para el masterplan

La Fase 1 no se cierra hasta resolver la capa NVIDIA. La capa AMD está
validada. Próximo paso: probar la Opción 1 (shim) en un job CUDA; si falla,
Opción 2.
