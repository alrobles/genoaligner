# Resumen de hallazgos: Neighbor Joining en GPU y exploración con geometría diferencial

Sesión Devin 2026-09-13, repo `alrobles/genoaligner-devel`. PRs: #1 (cerrado, hipótesis
sparse rechazada), #2 (NJ0, mergeado), #3 (A100, mergeado), #4 (L40/V100, este).
Documentos de detalle: `SPEC_NJ_GPU.md`, `RESULTADO_NJ_TRACE.md`,
`RESULTADO_NJ_GPU0.md`, `LITERATURA_NJ_GPU.md`, `DRAFT_NJ_GPU_ORIGINAL.md`.

## 1. Qué se entregó

`nj_tree_gpu(dist_packed, n, err, stats)` en `src/msa/nj_gpu.cpp` +
`include/genoaligner/backend/nj_kernel_impl.hip`: Neighbor Joining exacto denso en
device, **bit-exact con la referencia host `nj_tree`**, integrado como árbol guía por
defecto en `genomsa` (fallback explícito a `nj_tree_mt` ante cualquier error de device,
`GENOMSA_NJ=cpu` para forzar CPU). Tests sintéticos de paridad, `nj_bench`, y un sbatch
que valida sobre los 31 genes reales en ROCm o CUDA con el mismo código fuente.

| GPU | Backend | Paridad 31 genes | E2E COI | CYTB kernels (s) | Job |
|-----|---------|------|------|------|-----|
| MI210 | hipcc/ROCm 6.4.3 | 31/31 | byte-idéntico | 1.47 | 29238575 |
| A100-PCIe-40GB | nvcc 12.4 (sm_80) | 31/31 | byte-idéntico | 1.26 | 29238597 |
| V100-SXM2-16GB | nvcc 12.4 (sm_70) | 31/31 | byte-idéntico | 1.54 | 29239539 |
| L40 | nvcc 12.4 (sm_89) | pendiente (job encolado, nodo L40 ocupado) | | | 29239538 |

Referencia: CYTB (n=3523) tardaba 60 s en host 1 hilo, 9 s con `nj_tree_mt` 16 hilos.

## 2. Hallazgos técnicos

1. **La cereza NJ no es un par "cercano".** Medido con `tools/nj_trace.cpp` sobre 19 094
   rondas reales: en CYTB la cereza está en el top-32 por distancia de sus extremos sólo
   el 90 % de las rondas, y hay rondas donde no está ni en el top-128. En rondas tardías
   `r_i + r_j` domina a `(m-2)d_ij`. Esto invalida la idea central del draft (grafo de
   k vecinos por distancia + certificación por filas): la certificación barata cierra sólo
   el 2–6 % de rondas en los genes grandes, así que NJ1 costaría más que NJ0.
2. **Los empates de Q son reales (0.43 % de rondas, ≤ 4 ulp).** Un argmin paralelo con
   `atomicMin` sobre `q` cambia la topología. La reducción tiene que ser lexicográfica
   `(q, a, b)` reproduciendo "primer mínimo en orden `alive`" del host. Esto es lo que
   hace que el árbol GPU sea idéntico y no "equivalente".
3. **Exactitud FP64 cross-vendor es alcanzable y barata.** Tres reglas: sumar `r_i` en el
   mismo orden que el host (sin reducción en árbol ni atomics), `__dmul_rn` para impedir
   contracción FMA, y `d_new = 0.5*(d_iv + d_jv - d_ij)` en el mismo orden. Con esto
   MI210, A100 y V100 producen exactamente el árbol del host (y por tanto el mismo entre sí).
4. **El cuello es la latencia, no el ancho de banda.** rocprof: rowsum = 88 % del tiempo.
   Un hilo por fila con una cadena de sumas dependientes de longitud m. Pasar de leer
   `alive[b]` en global a tile en shared + unroll ×8 (más cargas en vuelo, mismo orden de
   sumas) dio ×64 (168 s → 2.6 s en CYTB) sin cambiar un bit. El ancho de bloque 64 dio
   otro ×1.8. Lo que queda (1.3–1.5 s) es la cadena de sumas en sí; la única vía exacta
   restante es más ILP (unroll 16, `double2`), no más paralelismo.
5. **Coste fijo de ~0.15–0.3 s por inicialización del runtime** (primer `hipMalloc`).
   En `nj_bench` aparece como `upload_s`; en `genomsa` no se paga dos veces. Para n < ~900
   `nj_tree_mt` sigue ganando en aislamiento; dentro del pipeline la GPU gana desde n≈300.
6. **Un `GPU Hang` único** (job v1, MI210) con kernels de 168 s encolados; no reproducido
   en 5 jobs posteriores en 4 GPUs con kernels < 1 ms. Hipótesis: watchdog de cola.

## 3. Geometría diferencial / hiperbólica: qué dice la evidencia

Bibliografía revisada (`LITERATURA_NJ_GPU.md` §6): Saitou–Nei, Semple–Steel, Felsenstein
1981; Dodonaphy (Soft-NJ), GeoPhy, VCSMC hiperbólico; Poincaré (Nickel–Kiela), HNN
(Ganea), Sarkar 2012; LKJ.

- **Soft-NJ (Dodonaphy)** sustituye el argmin por SoftSort: es una relajación
  diferenciable, no NJ. Viola el contrato de exactitud (I1) y no acelera nada por sí
  misma; su valor está en refinar el árbol guía por verosimilitud *después* de NJ0.
- **Embeddings hiperbólicos (Sarkar, Poincaré)** garantizan baja distorsión para
  *árboles*, no para matrices de distancia ruidosas de secuencias. La brecha "vecino
  cercano ≠ cereza NJ" que midió el tracer es precisamente esa distorsión. Un embedding
  hiperbólico como generador de candidatos (NJ1') sólo se justifica si demuestra
  Recall@32 > 0.95 con `nj_trace` — hoy los k-mer dan 0.90 en CYTB.
- **GeoPhy / VCSMC** pertenecen a la etapa de inferencia bayesiana/variacional sobre
  topologías, no a la construcción del árbol guía. Motivan usar el árbol NJ0 como
  inicialización de un espacio continuo.
- **LKJ**: la analogía del draft (parametrizar matrices válidas) no aplica: la
  restricción de un árbol es la aditividad (condición de cuatro puntos), no la
  positividad definida.

Conclusión: la geometría diferencial es la línea de *refinamiento* (árbol guía → espacio
hiperbólico → gradiente sobre verosimilitud), con NJ0 exacto como punto de partida
reproducible. No sustituye al kernel exacto.

## 4. Siguientes pasos recomendados

1. Extender `nj_trace` con un generador de candidatos por embedding hiperbólico
   (Sarkar sobre el árbol NJ parcial, o Poincaré sobre `d`) y medir Recall@32. Es un
   experimento CPU de una tarde y decide si NJ1' merece kernels.
2. Prototipo Soft-NJ/Dodonaphy en PyTorch sobre el árbol NJ0 de 2–3 genes, midiendo
   log-verosimilitud (Felsenstein) antes/después: cuantifica qué gana el refinamiento.
3. Rowsum: unroll 16 / `double2` manteniendo orden; medir en MI210 y A100.
4. Calidad biológica: comparar alineamientos `genomsa` (árbol GPU) vs MAFFT en los 31
   genes (`h14_vs_mafft.sbatch` ya existe) para cerrar la fila "calidad" del spec.
