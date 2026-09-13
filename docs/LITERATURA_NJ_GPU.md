# Literatura: Neighbor Joining paralelo / GPU y su relación con NJ0

Complemento de `docs/SPEC_NJ_GPU.md` y `docs/RESULTADO_NJ_TRACE.md`. Objetivo:
qué ha hecho la literatura para acelerar NJ, qué ideas son compatibles con el
contrato bit-exact de este repo (I1: `nj_tree_gpu ≡ nj_tree`), y qué queda
descartado o pospuesto.

## 1. Dos familias

| Familia | Idea | Exacta vs NJ canónico | Ejemplos |
|---|---|---|---|
| **Aceleración exacta** | Mismas O(n³) operaciones, ejecutadas en paralelo (GPU / multi-core / SIMD). | Sí si se controla suma y desempate. | Liu 2009 (CUDA), Zheng 2012 (CUDA), pNJTree 2006 (MPI), DIPPER 2026 (CUDA/HIP), DecentTree 2023 (OpenMP+SIMD). |
| **Poda de candidatos** | Evitar recorrer toda la matriz Q usando cotas / vecinos / ordenaciones. | Exacta sólo con cotas rigurosas (RapidNJ, QuickJoin, NINJA); heurística en FastNJ / relaxed NJ. | RapidNJ 2008, QuickJoin 2004, NINJA 2009, FastNJ 2008 (Elias–Lagergren), Clearcut. |

Nuestro experimento NJ1 (`RESULTADO_NJ_TRACE.md`) pertenece a la segunda
familia con certificación por filas; el gate falló (Recall@32 ≤ 0.93 y
≥ 94 % de rondas sin certificar en CYTB/COI). La literatura de la segunda
familia sigue siendo relevante para una fase posterior (§4), no para NJ0.

## 2. Aceleración exacta en GPU

### 2.1 Liu, Schmidt, Maskell — *Parallel reconstruction of neighbor-joining trees for large multiple sequence alignments using CUDA* (IPDPS 2009, DOI 10.1109/IPDPS.2009.5160923)
- Primer NJ en CUDA, motivado exactamente por nuestro problema: el árbol guía
  de MSA (ClustalW/MUSCLE) como cuello de botella para n grande.
- Matriz densa en device; por ronda: kernel de Q-mínimo (una fila por bloque,
  reducción intra-bloque) y kernel de actualización de la fila fusionada.
- Speedup reportado > 26× frente a CPU secuencial para datasets grandes.
- Lección para NJ0: el patrón "una fila `a` por bloque, hilos sobre `b`,
  reducción en shared memory, luego reducción final sobre m resultados" es
  el diseño base; el coste por ronda es O(m²) lecturas, coalescentes si la
  fila es contigua.

### 2.2 Zheng et al. — *Parallelization Mechanisms of Neighbor-Joining for CUDA Enabled Devices* (ChinaGrid 2012, DOI 10.1109/chinagrid.2012.32)
- Dos mecanismos: (a) paralelización por ronda (como Liu), (b) reducción de
  la dependencia entre rondas mediante granularidad dinámica (menos hilos
  cuando m es pequeño).
- Speedup medio 18.6× sobre miles de secuencias.
- Lección para NJ0: la cola de rondas pequeñas (m < ~256) domina el overhead
  de lanzamiento; considerar terminar en host cuando `m` cae bajo un umbral
  (I1 se conserva si el host ejecuta exactamente la misma aritmética).

### 2.3 DIPPER — Turakhia Lab, *Ultrafast and ultralarge distance-based phylogenetics using DIPPER* (Nat. Comput. Sci. 2026, DOI 10.1038/s43588-026-01015-8; código MIT: github.com/TurakhiaLab/DIPPER, CUDA y HIP/ROCm)
Implementación actual y abierta; `src/neighborJoining.cu` (305 líneas) es la
referencia más útil que hemos leído. Resumen del diseño:

- Matriz **densa n×n en `double`**, residente en device (n < 30 000 usa NJ
  convencional; por encima, placement / divide-and-conquer).
- Row sums `U` calculados **una vez** (`calculateU`, `atomicAdd` en shared) y
  después **actualizados incrementalmente** en el kernel de merge
  (`U[i] += new − old_x − old_y`).
- Argmin: `findMinDist` — cada hilo recorre un rango de filas × columnas,
  guarda su `(x, y, min)`; luego `thrust::min_element` sobre los
  `blocks×threads` candidatos comparando **sólo el valor**.
- Merge: `updateDisMatrix` — el nuevo nodo ocupa la fila/columna de `x`; la
  **última fila/columna se mueve a la de `y`** (swap-remove) para que la
  matriz viva quede contigua `[0, m−1)`. `realID[]` en host mapea slot→nodo.
- Sincronización host↔device por ronda (`cudaDeviceSynchronize` + 3
  `cudaMemcpy` de escalares) para calcular longitudes de rama en host.
- Criterio: `d_xy − U_x/(m−2) − U_y/(m−2)` (Q dividido por `m−2`, mismo argmin
  en exacto, pero **distinto redondeo** que `(m−2)d − r_x − r_y`).

Qué **no** cumple nuestro contrato y por qué importa:
1. `atomicAdd` y actualización incremental de `U` ⇒ orden de suma no
   determinista y distinto al host ⇒ rompe I1 (bit-exact) e incluso I2
   (determinismo run-a-run) cuando hay near-ties (0.43 % de rondas, tabla
   en `RESULTADO_NJ_TRACE.md`).
2. Desempate por orden de hilos, no lexicográfico `(a,b)`.
3. Swap-remove cambia el orden de `alive`, que en `nj_tree` define el orden
   de suma y el orden de recorrido del argmin.

Qué **sí** adoptamos: matriz densa `double` residente, nuevo nodo reutilizando
el slot de `x` (memoria n×n en vez de (2n−1)²), lista de merges en device para
evitar la sincronización por ronda, y soporte CUDA+HIP desde una sola fuente.

### 2.4 DecentTree — Wang, Barbetti, …, Minh (Bioinformatics 2023, btad536; github.com/iqtree/decenttree)
- CPU: NJ/BIONJ/UNJ vectorizados (SIMD) y paralelos (OpenMP), plantillas
  float/double, header-only; integrado en IQ-TREE 2.
- Incluye variantes "RapidNJ-style" (filas ordenadas + cota) además de las
  densas vectorizadas.
- Hasta 6× más rápido que RapidNJ en 64 000 genomas SARS-CoV-2.
- Lección: la versión densa vectorizada es competitiva hasta decenas de
  miles; la poda sólo paga en n ≫ 10⁴. Nuestro rango (n ≤ 3 523 hoy, ≤ 10⁴
  previsible) está en la zona densa. Referencia CPU natural para
  comparar NJ0 en tiempo absoluto (no en topología: DecentTree no garantiza
  nuestro desempate).

### 2.5 pNJTree — Du, Lin (Parallel Computing 2006, 32(5–6):441–446)
- NJ distribuido (MPI) para ClustalW; reparte filas de la matriz entre
  procesos, reduce el mínimo global y difunde la fila fusionada.
- Antecedente directo del esquema "row sums por fila + argmin por rango +
  broadcast del merge" que también usa `nj_tree_mt` en este repo.

### 2.6 *A GPU-based parallel method for evolutionary tree construction* (Comput. Electr. Eng. 2014, DOI 10.1016/j.compeleceng.2014.04.013)
- Revisión + implementación GPU; cita Liu 2009, RapidNJ y FastNJ. Confirma que
  la reducción del argmin y la actualización de la matriz son los dos kernels
  canónicos y que la dependencia estricta entre rondas es el límite de
  paralelismo (motivación original de NJ2, pospuesto).

## 3. Poda de candidatos (exacta y heurística)

- **RapidNJ** — Simonsen, Mailund, Pedersen (WABI/LNCS 2008). Cada fila
  ordenada por `d`; cota `Q_ab ≥ (m−2)·d_ab − r_a − r_max`; se recorre cada
  fila en orden hasta que la cota supera el mejor Q. Exacto. Coste práctico
  ~O(n²) en datos reales, pero la ordenación por fila se degrada con los
  merges (filas "sucias"); memoria ~2× la matriz. Nuestra certificación NJ1
  es exactamente esta cota aplicada a top-k; el resultado del tracer (sólo
  1.6–5.8 % de rondas cierran con k=32) indica que en nuestros genes la cota
  necesita recorrer una fracción grande de cada fila: `r_max` es demasiado
  laxo para distancias k-mer Jaccard, que están comprimidas en [0.6, 1.0].
- **QuickJoin** — Mailund, Pedersen (Bioinformatics 2004). Quad-tree sobre Q
  con cotas por región. Exacto; complejo de mantener en GPU.
- **NINJA** — Wheeler (WABI 2009). RapidNJ + external memory + filtro por
  cotas; exacto hasta 10⁵–10⁶ taxa en CPU.
- **Fast Neighbor Joining** — Elias, Lagergren (TCS 2008, 410:1993–2000,
  DOI 10.1016/j.tcs.2008.12.040). O(n²) mediante *visible set*; **no** es NJ
  canónico (reconstruye el mismo árbol sólo cuando la matriz es casi
  aditiva). Igual que *relaxed NJ* (Clearcut). Descartado por I1.

## 4. Implicaciones concretas para NJ0 (`nj_tree_gpu`)

1. **Layout**: `double` denso n×n row-major en device; nodo fusionado ocupa
   el slot de `xi`. Memoria 8n² bytes (CYTB n=3523 → 99 MB; n=10⁴ → 800 MB;
   MI210 64 GB permite n ≈ 80 000 antes de salir a NJ1/divide-and-conquer).
2. **`alive` en device, mismo orden que el host**: array de slots
   `alive[0..m)`; tras el merge `(bi, bj)`: compactación estable que elimina
   las posiciones `bi`, `bj` y añade el slot de `u` al final. Paralelo O(m),
   doble buffer.
3. **Row sums exactos**: `r[a] = Σ_{b≠a, b en orden alive} d[alive[a]][alive[b]]`,
   un hilo por `a`, suma **secuencial en el mismo orden** que el host. Nada
   de `atomicAdd` ni actualización incremental (diferencia 1 de §2.3). Se
   lee por columnas (`d` es simétrica) para que los hilos vecinos accedan a
   direcciones contiguas. Coste O(m²) por ronda con m-way paralelismo;
   aceptable para m ≤ 10⁴ (se mide en `RESULTADO_NJ_GPU0.md`).
4. **Argmin lexicográfico**: clave `(q, a, b)`, comparación
   `q < best || (q == best && (a < ba || (a == ba && b < bb)))`. Reproduce el
   "primer mínimo gana" del recorrido host `a<b`. Un bloque por `a`, hilos
   sobre `b>a`, reducción en shared, reducción final de los m parciales en el
   kernel de merge. Sin `atomicMin`.
5. **Aritmética idéntica al host**: `q = (double)(m−2) * d − r_a − r_b`,
   `d_new = 0.5 * (d_iv + d_jv − d_ij)`. Desactivar contracción FMA en device
   (`-ffp-contract=off` o `__dmul_rn`), porque el host x86-64 sin
   `-march=native` no fusiona y un FMA cambiaría el último bit de `q`.
6. **Sin sincronización por ronda**: los merges `(xi, xj)` se escriben en un
   array device `merges[round]`; el host descarga el array al final y
   construye `Tree` (nodos `n+round = {xi, xj}`, raíz = último). Coste host
   O(n). Esto elimina las 3 copias escalares + sync por ronda de DIPPER.
7. **Cola de rondas pequeñas** (§2.2): opcional terminar en host cuando
   `m ≤ M_host`, copiando la submatriz viva; se mide antes de activarlo
   (I1 se mantiene porque la aritmética es la misma).
8. **Fallback explícito**: cualquier `hipError_t` ⇒ `err` no vacío, `Tree`
   vacío, y el driver usa `nj_tree_mt`. Nunca un árbol aproximado silencioso.

## 5. Pospuesto / descartado

- NJ1 sparse-candidate: gate no superado (`RESULTADO_NJ_TRACE.md`). Podría
  revisarse con cotas por fila más ajustadas que `r_max` (RapidNJ usa
  `r_max` sobre la fila **ordenada**, que es más fuerte que top-k fijo).
- NJ2 merges batch por matching: requiere prueba de independencia de
  cerezas; sin literatura que lo haga exacto para NJ canónico.
- FastNJ / relaxed NJ / DIPPER placement: no son NJ canónico ⇒ violan I1.
- Soft-NJ / embeddings hiperbólicos / diferenciabilidad: fuera del alcance
  del árbol guía.
