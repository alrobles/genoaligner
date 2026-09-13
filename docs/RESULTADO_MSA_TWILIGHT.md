# genomsa vs TWILIGHT — benchmark y auditoría

Fecha: 2026-09-13. TWILIGHT 0.2.3 (TurakhiaLab/TWILIGHT, MIT, commit
f11db22). Comparación en el mismo hardware (KU HPC): V100-SXM2-32GB para
CUDA y MI210 gfx90a para HIP; genomsa compilado con nvcc sm_70 para la
comparación V100.

## 1. Calidad contra verdad simulada (scripts/msa_sim_truth.py)

n=64 hojas, ancestro 1500 bp, ~12% divergencia terminal, indels.
Column-SPS = fracción de pares homólogos verdaderos recuperados.

| alineador | árbol guía | SIM-SPS | ancho |
|---|---|---|---|
| TWILIGHT | NJ de genomsa | 0.9628 | 2436 |
| TWILIGHT | árbol verdadero | 0.9619 | 2429 |
| TWILIGHT -r 1 | NJ de genomsa | 0.9610 | 2398 |
| genomsa | NJ propio | 0.8964 | 2208 |
| (verdad) | — | 1.0000 | 2563 |

Lectura: con **el mismo árbol guía** TWILIGHT supera a genomsa por ~6.6
puntos de SPS — la brecha está en el motor (penalizaciones de gap
posición-específicas por columna + heurístico de columnas gappy), no en
la construcción del árbol. genomsa además sobre-compacta (2208 vs 2563
verdaderas): colapsa demasiados homólogos en matches. El árbol
verdadero no mejora a TWILIGHT (0.9619 vs 0.9628) — el cuello es el
motor, no la guía.

## 2. Velocidad (V100, mismo nodo)

| gen | n | genomsa | TWILIGHT |
|---|---|---|---|
| COI | 1608 | **14.5 s** | 54.8 s (88% perfiles diferidos a CPU) |
| CYTB | 3523 | 62.6 s | **10.4 s** |
| COI (MI210) | 1608 | ~11.6 s | 4.8 s (HIP) |

Velocidad depende del régimen; TWILIGHT escala mejor cuando cabe en
GPU, genomsa es competitivo en genes medianos.

## 3. Hallazgos de auditoría en TWILIGHT (verificados)

### 3a. Divergencia CUDA vs su propia referencia CPU (bug upstream)

COI, mismo árbol guía (COI.nwk), mismo binario:

| ruta | ancho |
|---|---|
| `twilight --cpu-only` (V100 o MI210, idéntico) | 2991 |
| HIP GPU (MI210) | 2991 — **byte-idéntico a CPU** |
| CUDA GPU (V100) | 5494, y 5382 en un re-run — **no determinista** |

- 1608/1608 filas difieren entre CUDA-GPU y CPU-ref.
- En MI210 solo 3/893 perfiles se difirieron; en V100 **786/893** (88%)
  cayeron a `fallback2cpu` porque `numBlocks` se dimensiona por
  `availableMem` (70% de la memoria libre del dispositivo).
- El camino diferido (`subtreeAln`/`mergeInsertions`) produce una
  alineación distinta del pipeline `--cpu-only` → el resultado final
  depende de la memoria de la GPU, no solo del input.
- El DP es entero (int16/int32): la no-determinación sugiere una race
  en la realineación diferida TBB, no ruido flotante.
- En sim64 (pequeño) CUDA==HIP==CPU byte-exacto: la divergencia solo
  emerge cuando el deferral se activa a escala.

### 3b. Fallback silencioso exclusivo de HIP

`src/hip/alignment-gpu.hip.cpp` (activo) vs `src/cuda/alignment-gpu.cu`
(comentado): cuando el path del kernel no consume las columnas
esperadas (`alnRef != hostLen[2n]`), HIP imprime `"CPU on No. X"` y
recalcula en CPU — encubre defectos del kernel HIP sin señal de fallo.
En nuestras corridas no se disparó (0 ocurrencias).

### 3c. Build roto con GCC 11 + nvcc

`-march=native` en CMAKE_CXX_FLAGS + headers AMX de GCC 11
(`amxtileintrin.h` usa `__builtin_ia32_*` que nvcc no define) → build
falla. Workaround: predefinir include guards
(`-D_AMXTILEINTRIN_H_INCLUDED` etc.) o quitar `-march=native`.
Además `-march=native` en un binario que corre en nodos
heterogéneos es frágil.

## 4. Oportunidades de fork/PR evaluadas

| candidato | evidencia | tamaño | valor upstream |
|---|---|---|---|
| Reporte + fix: deferral no determinista / memoria-dependiente | verificado (3a) | medio | alto — afecta reproducibilidad en GPUs pequeñas |
| Chequeo HIP→diagnóstico ruidoso o fix del kernel | verificado (3b) | pequeño | medio |
| Fix build `-march=native`/AMX | verificado (3c) | trivial | medio (portabilidad) |
| Test de paridad HIP↔CUDA↔CPU en CI | nuestro harness | medio | alto — nadie lo verifica hoy |
| DPX para sm_90+ opcional | código existe | n/a | bajo (ya tienen) |

Recomendación: abrir issue upstream con la reproducción 3a (datos
concretos, ambas GPUs) + PR con: fix de build (3c) y el diagnóstico
del fallback HIP (3b). El fix profundo del deferral requiere entender
por qué `mergeInsertions` difiere del pipeline CPU — más investigación.

## 5. Qué aprende genomsa de TWILIGHT

- Penalizaciones de gap **posición-específicas** por columna
  (`gapOpen[]`/`gapExtend[]` arrays) — probablemente la mayor parte
  del gap de calidad de 6.6 puntos.
- Heurístico `--remove-gappy` (columnas con >95% gaps no se alinean a
  fondo) — acelera sin costo de calidad (tw_nj vs tw_nj_r1 ≈ igual).
- DP bandeado con X-drop + checkpoints de convergencia → memoria de
  traceback constante (nosotros: O(m·n) direction bytes, 1.3 GB CYTB).
- Divide-and-conquer por sub-árboles (`--max-subtree`) para escala
  masiva.
