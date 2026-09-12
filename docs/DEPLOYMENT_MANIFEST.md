# Manifiesto de despliegue — llevar genoaligner a otro cluster

> **Entregable de Fase 8.** Este documento es **operativo**: dice qué copiar, qué
> re-verificar y qué puede fallar, con la evidencia de dónde salió cada afirmación.
> No es una lista de buenas intenciones.

---

## 0. Qué es genoaligner hoy (para que el lector no se llame a engaño)

- Una **librería C++ de alineamiento pairwise** por distancia de edición (WFA),
  con score y reconstrucción de CIGAR, que compila de **una sola fuente HIP** a
  **ROCm (hipcc)** y **CUDA (nvcc)**.
- **NO es un alineador múltiple.** No reemplaza a MAFFT/MACSE. Un pipeline que
  necesite MSA no lo puede usar como tal.
- **No hay API pública todavía**: los kernels se alcanzan desde los harnesses
  (`tests/parity/`, `bench/`). La exposición de API es lo que falta para uso real.

El valor demostrado es **portabilidad con paridad verificada**: los mismos números
en AMD y NVIDIA desde el mismo archivo, contra oráculos externos.

---

## 1. Requisitos, con lo que se midió en cada uno

### 1.1 Compiladores

    ROCm  ..... hipcc (wrapper de clang de AMD)  -> backend AMD
    CUDA  ..... nvcc + -D__HIP_PLATFORM_NVIDIA__ -> backend NVIDIA
    host  ..... g++ >= 12 si se usa CMake con gcc 14.2 (GLIBCXX_3.4.32)

**El mecanismo que hace posible la fuente única es una capa de headers dentro de
ROCm, no un shim instalable:**

    hip/hip_runtime.h
      #if defined(__HIP_PLATFORM_NVIDIA__) && !defined(__HIP_PLATFORM_AMD__)
        #include <hip/nvidia_detail/nvidia_hip_runtime.h>   ->  <cuda_runtime.h>

Consecuencia práctica: **el despliegue a NVIDIA necesita los headers de ROCm
presentes**, además de CUDA. No es "solo CUDA".

### 1.2 Versión de CUDA: la restricción que más veces costó tiempo

    CUDA 12.x  ->  la capa nvidia_detail de ROCm 6.4.3 compila
    CUDA 13.0  ->  NO compila (cudaMemLocation, cudaDeviceProp::clockRate, ...)

Y la arquitectura objetivo depende de la versión:

    sm_70 (V100) · sm_75 (q6000) · sm_80 (A100) · sm_89 (L40)  ->  nvcc 12.4 sirve
    sm_120 (PRO 6000)                                          ->  hace falta nvcc 12.8+

**12.8 es la única versión que satisface ambas condiciones** (conoce sm_120 y sigue
siendo 12.x). En KU está en un directorio versionado
(`/kuhpc/sw/nvhpc/Linux_x86_64/25.3/cuda/12.8`), no en los paths por defecto —
**recorrer todas las instalaciones versionadas antes de concluir que no existe.**

### 1.3 Hardware probado

    MI210 (gfx90a) · RTX 6000 (sm_75) · A100 (sm_80) · V100 (sm_70) · L40 (sm_89)
    · RTX PRO 6000 (sm_120)

Seis GPUs, dos vendors. Paridad 100% verificada contra el DP de CPU en todas.

---

## 2. Procedimiento de despliegue

### Paso 1 — Verificar el toolchain antes de compilar nada

```bash
# ¿Qué versiones de CUDA hay de verdad? RECORRER, no asumir.
# El glob necesita bash -lc explícito bajo ssh, o el * no se expande en el remoto.
ssh <host> 'bash -lc "for p in /path/*/cuda/*/bin/nvcc /path/cuda*/bin/nvcc; do
  [ -x \"\$p\" ] && echo \"\$p -> \$(\$p --version | grep -o \"release [0-9.]*\")\"; done"'
# ¿Qué arquitecturas soporta la que vas a usar?
nvcc --list-gpu-arch
# ¿El host g++ que nvcc aceptará? (nvcc rechaza > GCC 13 como host)
```

**Verificado en KU** (2026-09-12), el command exacto devuelve:

    /kuhpc/sw/nvhpc/Linux_x86_64/2021/cuda/11.4/bin/nvcc -> release 11.4
    /kuhpc/sw/nvhpc/Linux_x86_64/2024/cuda/12.4/bin/nvcc -> release 12.4
    /kuhpc/sw/nvhpc/Linux_x86_64/2025/cuda/13.0/bin/nvcc -> release 13.0
    /kuhpc/sw/nvhpc/Linux_x86_64/21.7/cuda/11.4/bin/nvcc -> release 11.4
    /kuhpc/sw/nvhpc/Linux_x86_64/24.11/cuda/12.6/bin/nvcc -> release 12.6
    /kuhpc/sw/nvhpc/Linux_x86_64/24.5/cuda/12.4/bin/nvcc -> release 12.4
    /kuhpc/sw/nvhpc/Linux_x86_64/24.7/cuda/12.5/bin/nvcc -> release 12.5
    /kuhpc/sw/nvhpc/Linux_x86_64/25.3/cuda/12.8/bin/nvcc -> release 12.8
    /kuhpc/sw/nvhpc/Linux_x86_64/25.9/cuda/13.0/bin/nvcc -> release 13.0

Un `find /` sobre el sistema de ficheros tarda >300s y expira: **usar los globs
versionados, no un find recursivo.** Y notar que los dos paths "por defecto"
(`2024`→12.4, `2025`→13.0) no incluyen el 12.8 que hace falta para sm_120.

Esta fase del despliegue es donde se pierde el tiempo si se salta: en KU, cuatro
rutas de instalación de un shim estaban cerradas y la solución era una versión de
CUDA que no estaba donde se buscó primero.

### Paso 2 — Compilar el backend AMD

```bash
hipcc -O2 -std=c++17 -I. -o /tmp/wfa_parity tests/parity/wfa_parity.cpp
```

### Paso 3 — Compilar el backend NVIDIA

```bash
R=/path/to/rocm            # headers de ROCm: IMPRESCINDIBLES
C=/path/to/cuda/12.4       # 12.4 para sm_70..sm_89; 12.8+ para sm_120
nvcc -w -D__HIP_PLATFORM_NVIDIA__ -std=c++17 \
     -gencode arch=compute_75,code=sm_75 \
     -I "$R/include" \
     -I "$C/include" -I "$C/targets/x86_64-linux/include" \
     -I . -lcuda -L "$C/lib64/stubs" \
     -x cu -o /tmp/wfa_parity_cuda tests/parity/wfa_parity.cpp
```

Flags que no son opcionales y por qué:
- `-x cu`: nvcc no reconoce la extensión `.hip`.
- `-lcuda` + `lib64/stubs`: **solo si el kernel consulta atributos de device**
  (`hipDeviceGetAttribute`), que la capa mapea a la API de driver. El kernel de
  traceback la necesita; el de score no.
- `-w`: la capa emite mucho ruido de deprecación; sin él los diagnósticos se
  entierran.

### Paso 4 — Correr el gate de paridad ANTES de confiar en nada

```bash
# MI210
sbatch scripts/h2_wfa_parity.sbatch      # espera paridad 100% y exit 0
# NVIDIA
sbatch scripts/h2_wfa_parity_cuda.sbatch
```

Criterio: `parity >= 95%` con **cero mismatch**, y `abandoned` contado aparte
(los pares con distancia real > smax son abandonados POR DISEÑO, no fallos).

### Paso 5 — Comparar contra oráculos externos (el brazo fuerte)

```bash
python3 -m pip install --user edlib rapidfuzz
python3 tests/parity/oracle_external.py --compare <emit.tsv>
```

Un DP propio comparte cualquier error conceptual propio. edlib/rapidfuzz son
implementaciones que no escribimos.

---

## 3. Verificación de replicabilidad

Correr el mismo check en cada host y **comparar versiones de toolchain**, no solo
veredictos. Idénticas versiones de hipcc/cmake en familias de OS distintas es la
evidencia real de que el entorno está fijado (ver `docs/CONTAINERS.md`).

    containers/build_image.sh compile
    containers/replicate_check.sh

La imagen `compile` no necesita GPU (lleva compiladores, no runtime), así que se
construye y prueba en cualquier CPU box — eso hace la replicabilidad medible.

---

## 4. Qué puede fallar al portar, y cómo se ve

| síntoma | causa | corrección |
|---|---|---|
| `undefined reference to cuDeviceGetAttribute` | kernel usa atributos de device | añadir `-lcuda` + `lib64/stubs` |
| `no member clockRate` en `nvidia_hip_runtime_api.h` | CUDA 13.0 con la capa 12.x | bajar a 12.x (12.8 si es sm_120) |
| `Don't know what to do with '.hip'` | nvcc no reconoce `.hip` | `-x cu` |
| `unsupported GNU version! gcc > 13` | nvcc y host compiler | `-ccbin /usr/bin/g++` |
| `relocation R_X86_64_32S ... PIE` | nvcc no emite PIC por defecto | `-Xcompiler -fPIE` + `-Xcompiler -pie` |
| kernels no corren: "no device" | job sin `--gres=gpu:<tipo>:N` | pedir la GPU en el sbatch |
| `module`/`ml` no existe en el job | Slurm no carga Lmod | `#!/bin/bash -l` o source del init |
| `Memory access fault` (rc=134) al lanzar | buffer de shared memory 1 int corto | revisar la aritmética de `2*smax+3` |

Estas siete no son hipotéticas: cada una costó al menos un job en el desarrollo.

---

## 5. Lo que el despliegue NO incluye todavía

- **API pública.** Hoy se usa desde los harnesses. Un consumidor externo necesita
  `include/genoaligner/api.hpp` (previsto en el masterplan §2.2, no escrito).
- **IO de FASTA.** `src/io/fasta.cpp` está en el plan, no existe.
- **Smith-Waterman.** Segundo método, no escrito.
- **Un benchmark de extremo a extremo con un usuario real.** Lo medido son los
  kernels, no un pipeline.

Declararlos aquí evita que un lector asuma que "desplegar" incluye usarlo.

---

## 6. Evidencia por afirmación

| afirmación | jobs / archivo |
|---|---|
| paridad 100% AMD y NVIDIA | 29184155, 29199289, 29201335, 29202787 |
| tres oráculos externos | `tests/parity/oracle_external.py` |
| traceback en ambos backends | 29207419, 29207420 |
| matriz de 6 GPUs | 29210660/61/62/64/65/29210709 |
| optimización del kernel (1.5-4.4x) | `docs/RESULTADO_FASE7_OPTIMIZACION.md` |
| build dual por CMake | `CMakeLists.txt`, jobs 29207287/29207294 |
| replicabilidad 3 hosts | `docs/CONTAINERS.md` |
