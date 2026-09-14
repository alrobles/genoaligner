# Resultado NJ0: Neighbor Joining exacto denso en GPU (`nj_tree_gpu`)

Rama `devin/1789341442-nj-gpu0`. Contrato: `docs/SPEC_NJ_GPU.md` (fase NJ0).
Jobs Slurm en KU HPC, partición `sixhour`, `gpu:mi210:1` (gfx90a), nodo
`r06r18n01`, 16 hilos host, ROCm 6.4.3: `29238546` (v1), `29238566` (v2,
rowsum pipelineado), `29238575` (v3, bloque rowsum 64). Script:
`scripts/nj_gpu_validate.sbatch`; datos por gen en
`/beegfs/a474r867/phylogenyAI/data/nj_gpu0/nj_gpu0_<job>.tsv`.

## 1. Veredicto

| Gate (SPEC §NJ0)                                            | Resultado |
|-------------------------------------------------------------|-----------|
| Paridad sintética en device (`tests/msa/test_nj_gpu.cpp`: n=1,2,3,4,5,8,17,33,64,1000; aleatorias, constantes, cero, cuantizadas con empates, métricas aditivas, tamaño empaquetado inválido) | PASS (3/3 jobs) |
| Paridad en los 31 genes reales: `Tree.nodes` y `root` idénticos a `nj_tree_mt` | PASS 31/31 (3/3 jobs) |
| End-to-end `genomsa` en COI: alineamiento byte-idéntico con árbol GPU vs `GENOMSA_NJ=cpu` | PASS (jobs 29238566, 29238575) — ver §4 sobre el job 29238546 |
| Fallback explícito: cualquier `hipError_t` ⇒ `err` no vacío, sin árbol | implementado; sin fallbacks observados (`exact_fallbacks=0`) |

NJ0 cumple I1 (bit-exact con la referencia) en hardware real. El árbol guía de
CYTB (n=3523) pasa de 60 s (host 1 hilo, `RESULTADO_MSA_GPU2.md`) / 9.0 s
(`nj_tree_mt`, 16 hilos) a **1.5 s de kernels + 0.3 s de upload** (×6 frente a 16 hilos, ×40 frente al host original).

## 2. Rendimiento (job 29238575, MI210)

`nj_gpu_s` incluye la conversión float→double densa en host, `hipMalloc`,
copias y la reconstrucción del árbol. `upload_s` (~0.27 s en todos los genes)
está dominado por la inicialización del runtime HIP en el primer `hipMalloc`
del proceso, no por la copia (la matriz de CYTB son 99 MB ≈ 0.05 s a PCIe).
En el driver `genomsa` el runtime ya está inicializado por el cálculo de
distancias, así que ese coste no aparece dos veces.

| Gen   | n    | `nj_tree_mt` 16T (s) | `nj_tree_gpu` total (s) | rounds (s) | upload (s) | device (MB) | paridad |
|-------|------|------|-------|-------|-------|------|------|
| CYTB  | 3523 | 9.05 | 1.83  | 1.47  | 0.30 | 99.4 | PASS |
| COI   | 1608 | 1.38 | 0.56  | 0.23  | 0.28 | 20.8 | PASS |
| IRBP  | 1252 | 0.89 | 0.45  | 0.12  | 0.27 | 12.6 | PASS |
| ND1   | 941  | 0.59 | 0.40  | 0.07  | 0.27 | 7.1  | PASS |
| BRCA1 | 913  | 0.56 | 0.40  | 0.07  | 0.28 | 6.7  | PASS |
| BMI1  | 140  | 0.07 | 0.34  | 0.003 | 0.28 | 0.2  | PASS |

Los 31 genes están en el TSV; todos PASS. Para n < ~900 el coste fijo de
inicialización domina y `nj_tree_mt` sigue siendo más rápido en aislamiento;
dentro de `genomsa` (runtime ya caliente) la GPU gana a partir de n ≈ 300.

### 2.1 Perfil por kernel (rocprof, COI, job 29238566)

| Kernel              | llamadas | total (ms) | media (µs) | %    |
|---------------------|----------|------------|------------|------|
| `nj_rowsum_kernel`  | 1606     | 308        | 192        | 88.1 |
| `nj_argmin_kernel`  | 1606     | 22         | 14         | 6.4  |
| `nj_merge_kernel`   | 1606     | 20         | 12         | 5.6  |

El rowsum domina porque el contrato exige sumar cada fila **en orden host**
(cadena de dependencias de longitud m por hilo, sin reducción en árbol ni
`atomicAdd`). El argmin y el merge son ya despreciables.

## 2.2 NVIDIA A100 (job 29238597, backend CUDA)

Mismos fuentes `.hip` compilados con `nvcc` 12.4 (`-x cu`, capa
`nvidia_detail` de ROCm, `sm_80`), nodo `r13r06n01`, A100-PCIE-40GB, 16 hilos
host. `sbatch --gres=gpu:a100:1 --export=ALL,BACKEND=cuda scripts/nj_gpu_validate.sbatch`.

| Gate | Resultado |
|------|-----------|
| Paridad sintética en device | PASS |
| 31 genes: `Tree.nodes`/`root` == `nj_tree_mt` | PASS 31/31 |
| E2E `genomsa` COI, árbol GPU vs `GENOMSA_NJ=cpu` | PASS (byte-idéntico) |

| Gen   | n    | `nj_tree_mt` 16T (s) | `nj_tree_gpu` total (s) | rounds (s) | upload (s) | MI210 rounds (s) |
|-------|------|------|------|-------|-------|-------|
| CYTB  | 3523 | 6.63 | 1.61 | 1.26  | 0.26  | 1.47 |
| COI   | 1608 | 1.12 | 0.48 | 0.20  | 0.18  | 0.23 |
| IRBP  | 1252 | 0.76 | 0.34 | 0.13  | 0.15  | 0.12 |
| ND1   | 941  | 0.50 | 0.24 | 0.06  | 0.13  | 0.07 |
| BRCA1 | 913  | 0.48 | 0.29 | 0.08  | 0.15  | 0.07 |
| BMI1  | 140  | 0.06 | 0.20 | 0.003 | 0.13  | 0.003 |

La paridad con la referencia host en ambos vendedores implica que el árbol
MI210 y el A100 son también idénticos entre sí (contrato I1, `SPEC_NJ_GPU.md`):
la suma en orden host y `__dmul_rn` evitan cualquier diferencia de contracción
FMA. Rendimiento equivalente (A100 ~15 % más rápido en CYTB, dominado por la
cadena de sumas dependientes del rowsum en ambos casos). TSV completo:
`nj_gpu0/nj_gpu0_29238597_cuda.tsv`.

## 2.3 NVIDIA V100 (job 29239539) y L40 (job 29239538)

Mismo script, `BACKEND=cuda`. V100-SXM2-16GB, `sm_70`, nodo con 16 hilos host:
sintético PASS, **31/31 genes idénticos**, E2E COI byte-idéntico.

| Gen   | n    | `nj_tree_mt` 16T (s) | `nj_tree_gpu` total (s) | rounds (s) | upload (s) | A100 rounds | MI210 rounds |
|-------|------|------|------|-------|-------|-------|-------|
| CYTB  | 3523 | 7.12 | 1.73 | 1.54  | 0.17  | 1.26 | 1.47 |
| COI   | 1608 | 1.27 | 0.42 | 0.30  | 0.10  | 0.20 | 0.23 |
| IRBP  | 1252 | 0.88 | 0.28 | 0.16  | 0.10  | 0.13 | 0.12 |
| ND1   | 941  | 0.59 | 0.20 | 0.07  | 0.10  | 0.06 | 0.07 |
| BRCA1 | 913  | 0.56 | 0.20 | 0.08  | 0.09  | 0.08 | 0.07 |
| BMI1  | 140  | 0.07 | 0.11 | 0.003 | 0.09  | 0.003| 0.003|

Tres GPUs de dos vendedores y tres generaciones (gfx90a, sm_70, sm_80) dan el
mismo árbol que el host en los 31 genes; el tiempo de kernels varía < 25 %
porque está dominado por la latencia de la cadena de sumas, no por el ancho
de banda ni el número de CUs/SMs.

L40 (`sm_89`): job 29239538 encolado (`--gres=gpu:l40:1`, un solo nodo L40 en
`sixhour`, ocupado; sin estimación de inicio al cierre de la sesión). Log en
`/beegfs/a474r867/genoaligner/logs/njgpu_29239538.{out,err}`; TSV en
`nj_gpu0/nj_gpu0_29239538_cuda.tsv` cuando termine.

## 3. Historia de la optimización (misma semántica, misma paridad)

| Versión | CYTB rounds (s) | COI rounds (s) | Cambio |
|---------|-----|------|--------|
| v1 (29238546) | 168.3 | 33.0 | rowsum: 1 hilo/fila, bucle escalar con `alive[b]` leído de global en cada iteración ⇒ latencia serializada (~7 µs·m por ronda) |
| v2 (29238566) | 2.64  | 0.35 | rowsum: `alive` por tile en shared, cuerpo desenrollado ×8 (8 cargas en vuelo, sumas en orden) |
| v3 (29238575) | 1.47  | 0.23 | bloque rowsum 256→64 para repartir los m hilos en más CUs |

v1→v2 es ×64 en CYTB sin cambiar un bit del resultado: el orden de las sumas
es idéntico, sólo se agrupan las cargas.

## 4. Incidencias

- Job 29238546, paso E2E: la primera ejecución de `genomsa` (árbol GPU) murió
  con `HW Exception by GPU node-4 ... reason: GPU Hang` antes de escribir el
  alineamiento; la ejecución con `GENOMSA_NJ=cpu` en el mismo nodo terminó
  bien, y en los jobs 29238566 y 29238575 (mismo nodo, mismo binario salvo el
  rowsum) el paso E2E pasó. No se ha reproducido. Hipótesis: el rowsum v1
  (168 s de kernels encolados sin sincronización en CYTB; ~33 s en COI)
  disparó el watchdog de la cola; v2/v3 mantienen cada kernel en <1 ms. No
  apareció tampoco en A100 (§2.2).
- hipcc emitió `-Wunused-result` en `hipFree` del destructor RAII; corregido.

## 5. Siguientes pasos

1. Reducir el coste fijo de `upload_s`: inicializar el runtime una vez por
   proceso (ya ocurre en `genomsa`); en `nj_bench` medirlo aparte.
2. Rowsum: probar 2–4 hilos por fila con **sumas parciales en orden** no es
   posible sin cambiar el orden de acumulación; la única vía exacta es más
   cargas en vuelo (unroll 16) o `double2` vectorizado sobre pares `(b, b+1)`
   manteniendo `s += v0; s += v1`.
3. Recoger el resultado L40 del job 29239538 (§2.3); A100 y V100 cerrados.
4. NJ1' (candidatos por embedding hiperbólico + certificación exacta): sólo
   si `tools/nj_trace.cpp` muestra Recall@32 > 0.95 (`LITERATURA_NJ_GPU.md` §6).
