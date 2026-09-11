# ROADMAP RETAKEOVER — genoaligner, camino a Fase 2

> **Repo:** `alrobles/genoaligner-devel` (privado) · **HEAD:** `5d76bc0`
> **Autor:** Ángel Luis Robles Fernández — Reuman Lab / KU
> **Fecha:** 2026-09-10 · **Versión:** 1.0
> **Complementa:** `docs/MASTERPLAN.md` (v1.0). No lo reemplaza.
> **Estado:** la formulación WFA está **resuelta y verificada al 100% en CPU**.

---

## 0. Resumen ejecutivo

El masterplan asumía que el kernel WFA era cuestión de "implementar la
formulación publicada". No lo era. Las cuatro versiones previas (v1-v4), el
harness de diagnóstico y el kernel HIP contenían **el mismo error estructural**,
y el error no era de dirección de diagonal ni de convención de coordenadas, sino
de **qué kernel de la referencia se estaba copiando**.

Estado real hoy:

    Formulación WFA en CPU : RESUELTA — 100% paridad (3000 pares + 1470 small)
    Kernel HIP             : TODAVÍA CONTIENE EL BUG (misma línea equivocada)
    H2 en MI210            : NO EJECUTADO (y no debía ejecutarse hasta ahora)
    Criterio Fase 2 (>=95%) : cumplido en CPU; no aún en GPU

El trabajo que queda es **portar una función ya probada**, no investigar más.
Ese es un cambio cualitativo: pasamos de "no sabemos si el álgebra está bien" a
"el álgebra está bien y hay que transcribirla".

---

## 1. Qué estaba mal (para no repetirlo)

### 1.1 La causa raíz, con evidencia

Cloné la implementación de referencia (`github.com/smarco/WFA`, MIT) y leí el
kernel real. `wavefront/wavefront_compute_edit.c` : `wavefront_compute_edit_idm`
(líneas 87-99), literal:

```c
const wf_offset_t ins   = prev_offsets[k-1];              // SIN +1
const wf_offset_t del   = prev_offsets[k+1];
const wf_offset_t misms = prev_offsets[k];
wf_offset_t max = MAX(del, MAX(ins,misms)+1);
if (WAVEFRONT_H(k,max) > text_length)    max = WAVEFRONT_OFFSET_NULL;
if (WAVEFRONT_V(k,max) > pattern_length) max = WAVEFRONT_OFFSET_NULL;
```

con (de `wavefront_offset.h`)

```c
#define WAVEFRONT_H(k,offset) (offset)        // coordenada de TEXT  (j)
#define WAVEFRONT_V(k,offset) (offset)-(k)    // coordenada de PATTERN (i)
```

Tres errores concretos, los tres medidos:

1. **Valor almacenado equivocado.** La referencia guarda la coordenada de TEXT
   `h` (= j). Los prototipos guardaban el índice de patrón `i`, o un "largo de
   diagonal", o un offset con otra definición. Mezclar esos tres hace que la
   extensión y el test de terminación no coincidan.

2. **`ins` NO lleva `+1`.** Este es el error caro. Yo lo tomé de
   `wavefront_compute_indel_idm` (línea 61 del MISMO archivo), que sí lo lleva
   — pero es un kernel para un modelo de coste distinto. Ese `+1` espurio
   inflaba cada inserción. (Nota de vergüenza: en un momento escribí `+1` en las
   tres transiciones, y en otro en ninguna; ambos estaban mal.)

3. **La asimetría es el algoritmo.** `MAX(del, MAX(ins,mism)+1)`: el `+1` vive
   *dentro* del grupo `(ins,mism)`, `del` queda fuera. Esa agrupación exacta es
   el contenido de WFA.

### 1.2 Dos trampas de implementación (encontradas, no supuestas)

- **El centinela se corrompe.** `WAVEFRONT_OFFSET_NULL = INT32_MIN/2`, y
  `NULLOFF + 1 != NULLOFF`. Un test de igualdad `offset == NULL` deja pasar
  basura a la extensión. Lo probé con un stack trace: el índice que reventaba
  era `18446744072635809796` (≈ -2^31). La referencia no lo sufre porque su
  centinela nunca se contamina; hay que usar un test de umbral
  (`offset > NULLOFF + margen`), no de igualdad.

- **El chequeo de límites es estricto y va antes de extender.**
  `h > text_length` (no `>=`), aplicado en el pase de cómputo; la extensión es
  un **pase separado** (`wavefront_extend_matches_packed_end2end`).

### 1.3 Traza empírica (números reales, todo en CPU local, segundos)

| Versión | Recurrencia | Paridad | Nota |
|---|---|---|---|
| v4 (la del plan) | `ins` sin `+1`, guarda `i` | **55.37%** | 1339/3000 fallos |
| v5 | `+1` en las tres | 90.83% | mejor, aún mal |
| v7 | `+1` en ninguna | 56.23% | y ahora sobrepasa (wfa>dp) |
| búsqueda por fuerza bruta | 32 variantes | 87.06% | así detecté que `ins` estaba mal |
| **v9 (port literal)** | **referencia verbatim** | **100.00%** | 3000/3000 + 1470/1470 |

Reproductor mínimo del bug original: `P=GG` vs `T=TTT` → daba 2, el correcto es 3.

**Lección de método, concreta:** el prototipo CPU no era "casi correcto". Era un
algoritmo distinto al de la publicación. Compilarlo y esperar ≥99%
(`/tmp/wfa_v4.cpp`) era una premisa falsa, no un plan. La corrección real fue
dejar de deducir el álgebra y portar la referencia. Los tres jobs de cluster
gastados antes no fueron el problema de método principal — el problema fue
tratar una implementación no verificada como base.

---

## 2. Principio rector desde aquí

    Ninguna línea corre en el cluster hasta que su gemela CPU dé 100% local.

Y una regla nueva, aprendida hoy y más importante que la anterior:

    Para cualquier kernel que reimplemente una formulación publicada,
    PORTAR LA REFERENCIA con cita de archivo:línea. No deducirla.

El coste de clonar la referencia fue ~1 minuto. El coste de no hacerlo fueron
cuatro versiones fallidas y tres jobs.

---

## 3. Fases de la retoma

### Fase R0 — Cerrar la formulación en CPU ✅ HECHA

Entregable: `/tmp/wfa_v9.cpp`, 100% paridad.
Evidencia: 16 casos a mano + 3000 aleatorios (3 regímenes) + 1470 exhaustivos,
0 fallos, 0 abandonados.

**Criterio de fusión (cumplido):** `EXIT=0` y `PARITY 100%`.

---

### Fase R1 — Portar al kernel HIP (bloqueante, ~1h)

El kernel `include/genoaligner/backend/wfa_kernel.hip` tiene el **mismo error**
que v4 (headers del archivo citan "v3" y explican una formulación equivocada).
Hay que reescribir su sección de cómputo para que sea la de v9.

Cambios concretos, línea por línea:

1. Reemplazar la sección "FORMULATION v3" del comentario por la de v9, citando
   `wavefront_compute_edit.c:87-99` y `wavefront_offset.h:52-53`.
2. `ins = prev[(k-1)+smax]` **sin `+1`**; `del = prev[(k+1)+smax]`;
   `mism = prev[k+smax]`; `max = max(del, max(ins,mism)+1)`.
3. Añadir el guard de centinela por umbral **antes** de derivar `(i,j)`.
4. Sustituir `o >= m && (o-k) >= n` por el test de la referencia:
   `k == n-m && offset >= n`, con `H=offset`, `V=offset-k`.
5. `if (threadIdx.x == 0)` para el escaneo de terminación — mantener, pero
   recordar que el hilo 0 debe escanear solo `[-s,s]`.
6. Revisar el `shmem` y el mapeo `k = tid - s`: el bloque debe cubrir `2*smax+1`
   diagonales (el harness ya lo documenta en su comentario de launch).

**Criterio de fusión:** el mismo binario fuente compila con `hipcc` y el caso
`P=GG T=TTT` devuelve 3 en GPU, y `P=C T=GT` devuelve 2.

---

### Fase R2 — Paridad GPU vs CPU en MI210 (H2, ~30 min de job)

**Antes de lanzar:** verificar que el harness `tests/parity/wfa_parity.cpp`
compare en GPU↔CPU sobre el mismo set determinista (ya lo hace, seed 12345).

Un bug adicional detectado en el harness, **corregir primero:**

```cpp
// línea 158-160: los punteros se codifican como OFFSETS en el campo puntero
views[i].text    = (const char*)(ot);   // offset host
views[i].pattern = (const char*)(op);
```

Esto funciona por accidente (el offset se reinterpreta luego), pero es UB y no
es lo que el kernel espera. Además: `views[i].text = d_text + (size_t)v.text;`
reinterpreta un offset de 64 bits guardado en un puntero — en CPU local puede
pasar, en GPU con `__restrict__` es frágil. Cambiar a un campo `int32_t` de
offset explícito y resolver a puntero de dispositivo antes del `hipMemcpy`.

**Criterio de fusión (Fase 2 del masterplan):** ≥95% sobre ~1000 pares.
Con v9 portado esperamos 100%, y de hecho R2 es una *confirmación*, no una
exploración.

Comando (ya existe):

    sbatch scripts/h2_wfa_parity.sbatch

Recordatorio de entorno (de la skill `ku-hpc-gpu-toolchain`):
- `#!/bin/bash -l` obligatorio (los jobs Slurm no cargan Lmod).
- build dentro de Apptainer (`genoaligner-compile.sif`), sin root.
- `scontrol` para medir el pool, nunca `sinfo`.

---

### Fase R3 — Preservar el harness de diagnóstico (rápido, importante)

Tu punto 2 es correcto y lo subo a fase propia porque el valor es permanente.

Mover `/tmp/wfa_diag.cpp` → `tests/parity/wfa_diag.cpp` como test **separado**
del de paridad, con un propósito explícito: **separar fallos por régimen**
(subs / indels / mixtos). Fue exactamente eso lo que convirtió un "90% opaco" en
un bug localizado. En el camino de hoy, los tres regímenes también mostraron
patrones distintos (55% global pero subs 60/1000 vs mixed 167/1000), que es una
pista que un test agregado habría escondido.

Además: añadir a `tests/parity/` los **casos mínimos** como regresión fija:
`GG/TTT`, `CT/CCC`, `CG/GAC`, `C/GT`, más `A/AA` (el que reventaba por centinela).
Son cuatro líneas de tabla y bloquean la reintroducción exacta de cada bug.

**Criterio de fusión:** `ctest` (o `make test`) falla si se reintroduce cualquiera
de los cuatro bugs.

---

### Fase R4 — Commit y cierre de Fase 1 pendiente

Tu punto 3. El repo tiene 5 untracked y **cero commits** desde `5d76bc0`.
Contenido a commitear, en commits atómicos:

    docs/ROADMAP_RETAKEOVER.md        (este archivo)
    src/reference/edit_distance_cpu.hpp
    tests/parity/wfa_diag.cpp         (desde /tmp, Fase R3)
    tests/parity/wfa_parity.cpp
    include/genoaligner/backend/wfa_kernel.hip   (tras R1)
    scripts/h2_wfa_parity.sbatch
    containers/genoaligner-nvidia-shim.def

Mensajes sugeridos (uno por commit, sin aplastar):

    wfa: fix formulation — port reference kernel verbatim (100% CPU parity)
    wfa: harden sentinel check; boundary test is strict (h > n)
    tests: promote wfa_diag to a permanent per-regime regression test
    tests: add minimal reproducers GG/TTT, CT/CCC, CG/GAC, C/GT, A/AA
    h2: fix PairView offset encoding before first GPU parity run
    h1c: record NVIDIA shim verdict (negative, exact cause)

**Sobre el `.def` de NVIDIA:** el veredicto negativo ya está explicado en
`5d76bc0` ("wrong base image"). La skill `ku-hpc-gpu-toolchain` §11 dice que
`hip-nvcc` no existe en PyPI y que el `hipcc` de conda-forge está congelado en
6.3.3. Es decir: la capa NVIDIA **no** se resuelve con el shim por esa vía.
Decisión pendiente (ver §5).

---

### Fase R5 — Cerrar la capa NVIDIA (decisión, no exploración)

Esto es lo único que sigue *abierto* del masterplan (Fase 1 no cierra sin ello).
Dado el hallazgo, hay tres opciones reales:

| Opción | Coste | Riesgo | Comentario |
|---|---|---|---|
| A. hipify + nvcc para NVIDIA | ~1d | Bajo | Cero deps nuevas. Rompe "una sola fuente HIP". |
| B. shim HIP compilado a mano | ~2-3d | Alto | El camino oficial, pero sin paquete disponible. |
| C. Aceptar AMD-only y diferir NVIDIA | 0 | Medio | Estratégicamente defendible (ver abajo). |

**Observación estratégica que cambia el cálculo:** el §10 del masterplan ya
dedujo que *la flota AMD es la única que compila HIP sin fricción, y las MI210
son el hardware que ninguna herramienta de alineación soporta*. El valor del
proyecto está en las MI210. La capa NVIDIA es **diferenciación de producto**
("portable"), no **necesidad técnica** (phylogenyAI corre en KU, en MI210).

Recomendación: **Opción A** si el objetivo es publicar con la afirmación
"portable CUDA+ROCm"; **Opción C** si el objetivo es destrabar phylogenyAI
primero. No decidir por defecto — decidirlo explícitamente.

---

## 4. Orden crítico (qué bloquea qué)

```
R0 CPU parity  ✅
   │
   ├─► R1 portar kernel ──► R2 H2 en MI210 ──► (Fase 2 masterplan ✅)
   │                                              │
   └─► R3 regression test ──► R4 commits ◄────────┘
                                   │
                                   └─► R5 decisión NVIDIA ──► cierra Fase 1
```

R1 es el único bloqueante real. R3/R4 son paralelizables con R1-R2.
R5 es una decisión, no trabajo de laboratorio.

---

## 5. Decisiones que necesito de ti (con pros/contras y datos)

Las tres primeras están en el §8 del masterplan y siguen abiertas; la cuarta es
nueva y la destapó el bug de hoy.

1. **Capa NVIDIA:** ¿A (hipify), B (shim a mano) o C (diferir)?
   Datos: shim no disponible por PyPI/conda-forge (skill §11); hipify-clang ya
   está en `/kuhpc/sw/rocm/6.4.3/bin/`. Mi lectura: A o C, nunca B.

2. **WFA estándar vs adaptive primero.** Sigue recomendado estándar — y ahora
   con más razón: la formulación estándar ya está probada. Adaptive es
   complejidad nueva sobre base nueva.

3. **CIGAR propia vs SAM-compatible.** Sin cambio respecto al masterplan.

4. **NUEVA — ¿el `.def` NVIDIA se queda en el repo o se archiva?**
   Contiene un enfoque que sabemos que no funciona (base image equivocada) y sin
   shim disponible. Opciones: (a) conservarlo con un README que documente el
   callejón sin salida, (b) moverlo a `docs/negative_results/`. Prefiero (a):
   los resultados negativos documentados son activo, y aquí el "por qué" ya está
   en el mensaje de commit. Tu llamada.

---

## 6. Lo que NO hay que hacer

- **No lanzar H2 sin portar R1.** Daría ~55% y volveríamos a leer un fallo
  opaco. El criterio ≥95% no cambia el hecho de que el kernel actual es el
  algoritmo equivocado.
- **No "arreglar" el kernel por prueba y error en el cluster.** El bug de hoy se
  resolvió en minutos en CPU porque cloné la referencia. Repetir el patrón de
  iterar en GPU es exactamente el error de método que este roadmap corrige.
- **No confundir este roadmap con progreso.** R0 está hecho; R1-R5 no. La
  formulación es la parte que parecía más difícil y resultó ser la que se
  resolvió; el port es mecánico pero **no está hecho**.

---

## 7. Próxima acción inmediata

**Ejecutar R1:** portar la recurrencia de `/tmp/wfa_v9.cpp` a
`include/genoaligner/backend/wfa_kernel.hip`.

Concretamente: reescribir el bloque de cómputo (líneas ~121-157 del kernel
actual) con `ins` sin `+1`, el test de terminación `k == n-m && h >= n`, y el
guard de centinela por umbral.

Coste: ~1h local (compilable con `hipcc --offload-arch=gfx90a` si hay ROCm
local; si no, validar la lógica con un shim de `hip/hip_runtime.h` en CPU antes
de tocar el cluster).

Después, y solo después: `sbatch scripts/h2_wfa_parity.sbatch`.

---

## 8. Resultados R1-R2 (2026-09-10/11)

### R1 — port del kernel ✅ COMPLETO

`include/genoaligner/backend/wfa_kernel.hip` reescrito como port verbatim de
`wavefront_compute_edit_idm`. Verificado por `scripts/check_kernel_cpu.sh`
(exit 0): 15/15 casos a mano, 3000/3000 en los tres regímenes, 1470/1470 pares
exhaustivos ≤6, contra el DP de `src/reference/edit_distance_cpu.hpp`.

### R2 — H2 en MI210 ✅ PASS

Dos submits, y el primero es el resultado más valioso de la fase.

**Intento 1 — job 29184154: FALLO (rc=134)**

    H2: FAIL (rc=134)
    Memory access fault by GPU node-4 (Agent handle: 0xf14bd0)
    on address 0x14af83abc000. Reason: Unknown.
    /tmp/wfa_parity 1000  ->  Aborted (core dumped)

...pese a que el CPU gate había pasado. **Causa: el buffer de wavefront estaba
un `int` corto.** El padding existía solo en el lado bajo. Aritmética con
`smax=64`:

    launch allocaba   2*(2*smax+1+1) = 260 ints
    B empieza en      wf_stride      = 130
    último toque de B  idx(+smax+1)   = 130 + 130 = 260
    máximo válido     259
    -> desbordamiento de exactamente 1 int

`idx(k-1)` en `k=-smax` necesita el slot 0, y `idx(k+1)` en `k=+smax` necesita
el slot `2*smax+2`. **Los dos extremos necesitan un slot**: el tamaño por
wavefront es `stride+2`, no `stride+1`. Corregido en `eba5944`.

**Por qué el gate CPU no lo vio** (y la corrección estructural): la rama shim
del kernel hacía `shim::smem_vec().assign(total, ...)` por su cuenta, así que el
harness nunca controlaba la asignación y una escritura fuera de rango caía
dentro de un buffer mayor. El kernel ya no re-asigna; el harness dimensiona el
buffer exactamente como el launch site, de modo que un launch corto es ahora un
error de ASan local. El gate ganó además un pase `asan+ubsan`.

Lección: **el gate CPU es necesario pero no suficiente.** No ve sincronización,
warp behaviour, ni límites de asignación de device. Este bug lo encontró el
hardware, no el razonamiento.

(Aparte: el probe de sanitizers del gate era por preprocesado, y GCC 11.5 del
login node tiene headers pero no `libasan.so.6.0.0`; eso producía un "fallo de
memoria" fantasma que bloqueaba un submit válido. Corregido en `1cd727f`: el
probe ahora compila y enlaza.)

**Intento 2 — job 29184155: PASS**

    === host: r06r18n01 ===
    HIP version: 6.4.43484-123eb5128
    genoaligner WFA parity harness (H2)
      device : AMD Instinct MI210
      arch   : gfx90a:sramecc+:xnack-
      cases  : 1000
      smax   : 64
      block  : 256 threads (need 129 for 2*smax+1 diagonals)

    === H2 RESULT (WFA score parity) ===
      resolved  : 946
      pass      : 946
      fail      : 0
      abandoned : 61  (edit distance > smax=64)
      parity    : 100.00%  (criterion: >= 95%)
      H2: PASS

**Criterio de fusión de Fase 2 (≥95%): CUMPLIDO, con 100%.**

Nota sobre los 61 abandonados: son pares cuyo `isD > smax=64` (los regímenes
`len=256` del set de control), no fallos. El harness los excluye correctamente
del denominador. Subir `smax` los resolvería, a costa de más memoria compartida
por bloque.

### Comandos reproducibles

    # gate local / login node (sin GPU)
    bash scripts/check_kernel_cpu.sh

    # pre-flight de entorno (imagen + GPU visibles)
    sbatch /beegfs/a474r867/genoaligner/scripts/smoke_mi210.sbatch

    # paridad real
    sbatch scripts/h2_wfa_parity.sbatch

### Estado de las fases

    R0  paridad CPU         HECHO
    R1  port al kernel HIP  HECHO — 100% CPU, exit 0
    R2  H2 en MI210         HECHO — 100%, job 29184155
    R3  tests de regresión  HECHO
    R4  commits             HECHO
    R5  capa NVIDIA         HECHO — 100% CUDA, job 29199289
    R6  Fase 3 externa      HECHO — 100% vs 3 oráculos, job 29201335

### R6 — Fase 3: paridad contra oráculos externos ✅ CERRADA

**Resultado: la GPU coincide al 100% con TRES implementaciones que no
escribimos.** Job 29201335 en MI210:

    cases      : 1007
    resolved   : 946
    agree      : 946
    mismatch   : 0
    abandoned  : 61   (isD > smax=64; no es fallo)
    parity     : 100.00%
    seqan3     : 946 filas verificadas, 0 desacuerdos

    edlib (Myers bit-vector)  ·  rapidfuzz (bit-parallel Levenshtein)  ·  SeqAn3 3.4.0

Las tres se verifican entre sí en cada par; si dos discrepan, el oráculo falla
en vez de emitir un número. En 300 pares de control: cero desacuerdos entre las
tres.

**Por qué esto es más fuerte que H2.** H2 comparó la GPU contra nuestro propio
DP: ambos lados comparten cualquier error conceptual nuestro. Contra
implementaciones externas, ese riesgo compartido desaparece.

**Corrección importante sobre SeqAn.** En R5 afirmé que SeqAn era "un proyecto
C++ aparte" y lo descarté sin medirlo. **Era falso.** SeqAn3 es header-only:
`git clone --depth 1` = 4.3 MB en segundos, y compila con
`/kuhpc/sw/gcc/14.2/bin/g++ -std=c++23`. Coste real: ~25 minutos, de los cuales
5 fueron intentos de compilación por errores de API. Parasail, en cambio, sí
resultó inservible (binding roto para coste unitario).

Otra vez el mismo patrón: **deduje la API en vez de leer la fuente autoritativa.**
Con SeqAn el snippet funcional estaba dentro del propio header
(`align_pairwise.hpp`). Dos de mis afirmaciones en los comentarios del kernel y
una en el plan fueron falsas por esta razón.

**Un bug encontrado al probar el test, no el código:** `--compare` imprimía
PASS sobre **cero casos resueltos**. Silencio leído como éxito, el peor modo de
fallo posible para una herramienta de verificación. Ahora devuelve 2
(INCONCLUSIVE). Lo destapó pasar el test contra un fichero vacío en vez de
confiar en que un test que pasa significa que el test funciona.

**Pipeline reproducible:**

    sbatch scripts/h3_external_parity.sbatch     # GPU + 3 oráculos, autocontenido
    python3 tests/parity/oracle_external.py --compare <emit.seqan3.tsv>
    bash tests/parity/e2e_fase3.sh               # valida que el oráculo PUEDE fallar

### R5 — capa NVIDIA ✅ CERRADA, opción A

**Resultado: la portabilidad CUDA funciona, sin hipify y sin dependencia nueva.**
La misma fuente `wfa_kernel.hip` da 100% en ambos backends:

    MI210   (gfx90a, hipcc 6.4.3) ........ 100.00%  job 29184155
    RTX 6000 (Turing sm_75, nvcc 12.4) ... 100.00%  job 29199289
    mismos números: 946 pass / 0 fail / 61 abandoned

**El hallazgo que cambia el diagnóstico.** H1 había concluido que la capa NVIDIA
era imposible, porque buscó un *shim instalable* y no lo encontró (cuatro rutas
cerradas, todas reales). La conclusión era falsa por un motivo de categoría: el
mecanismo no es un paquete, es **una capa de headers que ya estaba en el
cluster**.

    hip/hip_runtime.h
      #if defined(__HIP_PLATFORM_NVIDIA__) && !defined(__HIP_PLATFORM_AMD__)
        #include <hip/nvidia_detail/nvidia_hip_runtime.h>  ->  <cuda_runtime.h>
      #elif defined(__HIP_PLATFORM_AMD__) ...
      #else #error "Must define exactly one of ..."

El compilador para NVIDIA es `nvcc`, no hipcc. Lo que hace viable la fuente única
es esa capa más los include paths:

    nvcc -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
         -gencode arch=compute_75,code=sm_75 \
         -I /kuhpc/sw/rocm/6.4.3/include \
         -I $CUDA/include -I $CUDA/targets/x86_64-linux/include \
         -I . -x cu -o wfa_parity_cuda tests/parity/wfa_parity.cpp

**Restricción medida:** CUDA **12.4**, no 13.0. El layer `nvidia_detail` de ROCm
6.4.3 apunta a 12.x; con 13.0 falla en `cudaMemLocation` y en
`cudaDeviceProp.clockRate/.computeMode`. Usar
`/kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4`.

**Coste real:** ~2 horas, de las cuales la mayor parte fue diagnóstico. Cero
traducción. La opción A estimada en "~1 día" quedó obsoleta.

**Corrección al `MASTERPLAN` §2.1:** decía `hipcc con -D__HIP_PLATFORM_NVIDIA__`,
que es engañoso (hipcc es un wrapper de clang de AMD y no envuelve a nvcc). Ya
corregido en el propio masterplan.

**Secuencia de diagnóstico, para el registro:** hipify-clang falló por headers
cuRAND ausentes en el nvhpc del sitio (no es un problema de flags); hipify-perl
dio un diff de **cero reescrituras** (nuestro código usa solo API compartida);
y la prueba directa de `nvcc` compiló a la primera. Los tres resultados juntos
son los que cerraron la cuestión: el paso de traducción no solo era innecesario,
era imposible de ejecutar en este sitio y tampoco hacía falta.

**Lección de método:** "cuatro rutas agotadas con evidencia" en un skill seguía
siendo una hipótesis. Diez minutos de `nvcc` directo la desmintieron. Un negativo
bien documentado no es un hecho verificado hasta que lo reproduces en el caso
concreto que te importa.

### Nota de infraestructura: artefactos en beegfs

`images/genoaligner-compile.sif` (5.2 GB) vive en scratch y **no** es durable.
Receta de reconstrucción: `containers/build_image.sh compile`. Un `ls` filtrado
por `grep` reportó este artefacto como ausente durante esta sesión y estuvo a
punto de provocar un rebuild innecesario: **no canalizar un chequeo de
existencia por un filtro.**
