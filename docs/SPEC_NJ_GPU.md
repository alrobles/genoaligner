# SPEC — Neighbor Joining en GPU para genomsa (NJ-GPU)

> Estado: borrador v0.1 (2026-09-13). Contrato previo a escribir kernels.
> Origen: draft "GPU-Accelerated Differentiable Neighbor Joining for MSA
> Phylogenetics" (usuario) + evidencia de `RESULTADO_MSA_GPU2.md`.

## 0. Por qué

En el pipeline genomsa el kernel DP ya no es el cuello. Medido en MI210
(`results/msa_gpu/timings.tsv`, corridas previas a `_mt`):

| gen  | n    | dist (host) | **NJ (host)** | align (GPU) | total |
|------|------|-------------|---------------|-------------|-------|
| CYTB | 3523 | 35.8 s      | **60.4 s**    | 3.4 s       | 99.7 s |
| COI  | 1608 | 4.1 s       | **5.4 s**     | 1.9 s       | 11.5 s |
| BRCA1| ~700 | 1.1 s       | 1.0 s         | 12.6 s      | 14.8 s |

`nj_tree` es O(n³) con matriz densa `(2n-1)²` en `double` y recomputa los
row-sums en cada ronda. `nj_tree_mt` reparte filas entre hilos sin cambiar
la complejidad. A n≈3.5k el árbol guía cuesta ~18× el alineamiento.

## 1. Contrato

### 1.1 Interfaz

```cpp
namespace genomsa {
// Misma firma que nj_tree / nj_tree_mt. Misma semántica (§1.2).
Tree nj_tree_gpu(const std::vector<float>& dist_packed, int n,
                 std::string& err, NjStats* stats = nullptr);
struct NjStats {
    double upload_s = 0, rounds_s = 0, download_s = 0;
    int    rounds = 0;         // n-2 merges + raíz
    int    exact_fallbacks = 0; // NJ1: rondas que necesitaron reducción completa
    size_t dev_bytes = 0;
};
}
```

El driver `msa_align_gpu` la usa cuando `GENOMSA_NJ=gpu` (default: host
hasta pasar el gate §4). `tree_levels`, el merge y el DP no cambian.

### 1.2 Semántica de referencia (no negociable)

La referencia es `nj_tree` (`src/msa/msa_ref.cpp`):

1. Entrada: triángulo inferior empaquetado `float`, índice
   `i*n - i(i+1)/2 + (j-i-1)`, `i<j`. Trabaja en `double`.
2. Ronda con `m` nodos vivos: `r_a = Σ_{b≠a} d_ab` (suma secuencial en
   orden `alive`), `Q_ab = (m-2) d_ab - r_a - r_b`.
3. Argmin de Q sobre `a<b` en orden de escaneo `(a,b)`; **empate → primer
   par encontrado** (menor `a`, luego menor `b`, en índices de `alive`).
4. Nuevo nodo `u = |nodes|`; hijos `{alive[bi], alive[bj]}` en ese orden
   (izquierda = menor índice en `alive`).
5. `d_uv = 0.5 (d_iv + d_jv - d_ij)`; `alive := rest ++ [u]` (u al final).
6. Termina con 2 vivos → raíz `{alive[0], alive[1]}`.
7. `Tree.nodes[0..n-1]` hojas, internos `n..2n-2`, `root = 2n-2`.

### 1.3 Invariantes de aceptación

- **I1 (topología)**: `nj_tree_gpu(D,n).nodes == nj_tree(D,n).nodes`
  (igualdad de vectores, incluye orden de hijos y numeración de nodos)
  para todo `D` del set de gate (§4). Esto es más fuerte que RF=0: fija
  el orden de merge del que dependen `tree_levels` y la paridad byte a
  byte del MSA.
- **I2 (determinismo)**: dos corridas sobre el mismo input dan el mismo
  árbol, en el mismo dispositivo y entre MI210 / NVIDIA.
- **I3 (aritmética)**: la GPU reproduce el **orden de suma** de la
  referencia para `r_a`, o bien la referencia se redefine con un orden
  de reducción explícito (árbol binario por bloques) que ambas
  implementaciones comparten. **Decisión pendiente (D1, §6).** Mientras
  no se decida, el gate exige I1 y reporta el número de rondas con
  empate numérico de Q dentro de 1 ulp relativo (donde I1 podría
  romperse por reordenar sumas).
- **I4 (memoria)**: `dev_bytes ≤ 8·(2n-1)² + O(n)` en NJ0 (~400 MB a
  n=3523, cabe en cualquier GPU del cluster). NJ1 debe bajar a O(n·k).
- **I5 (fallo controlado)**: sin dispositivo, o si `n` excede memoria,
  `nj_tree_gpu` devuelve `false`/err y el driver cae al host. Nunca un
  árbol "aproximado" sin marcarlo.

## 2. Fases (test-and-drop, cada una con gate propio)

### NJ0 — exacto denso en GPU

Objetivo: eliminar el cuello sin cambiar el algoritmo. Complejidad O(n³)
sigue, pero el trabajo por ronda (Θ(m²) lecturas) es exactamente lo que la
GPU hace bien: a n=3523 son Σ m² ≈ 1.5·10¹⁰ lecturas de `double`, ~1–2 s
en MI210 por ancho de banda, frente a 60 s en host.

Por ronda, 3 kernels + 1 copia D2H de 8 bytes:

1. `nj_rowsum`: un bloque por fila viva, reducción de `m` elementos.
   Orden de suma: ver D1.
2. `nj_argmin_q`: reducción sobre los `m(m-1)/2` pares `(a,b)` con clave
   `(Q, a, b)` lexicográfica — reproduce el desempate "primer par en
   orden de escaneo" sin depender del orden de la reducción.
3. `nj_merge`: escribe la fila/columna `u`, compacta `alive` (swap-remove
   **no** vale: el orden de `alive` es parte de la semántica §1.2.5;
   se usa remove-preserving-order con un prefix-scan o se mantiene
   `alive` en host, O(m) por ronda, que es despreciable).

Matriz densa `(2n-1)²` en device, `alive` como lista de índices en device
(`m` enteros), 1 sincronización por ronda (leer `(bi,bj)`).

Gate NJ0: I1 sobre §4 en MI210 y en una NVIDIA; `rounds_s` reportado para
CYTB, COI, BRCA1 y `msa_gpu_validate.sbatch` sigue `PARITY: PASS` con
`GENOMSA_NJ=gpu`.

Kill: si I1 falla por sumas reordenadas y D1 no se resuelve con orden de
reducción compartido, la fase se cierra documentando el caso.

### NJ1 — sparse-candidate certificado

Hipótesis central del draft (§9): "la cereza NJ exacta está en un
vecindario top-k pequeño de sus extremos en la mayoría de las rondas".

Diseño: grafo de candidatos `E_k` (k vecinos más cercanos por nodo vivo,
según `d`), `Q` sólo sobre `E_k`, propuesta `(i,j) = argmin_{E_k} Q`, y
**certificación** antes de aceptar. Certificación por filas (nivel 3 del
draft): la propuesta es exacta si para cada nodo `a` vivo,
`min_{b∉E_k(a)} Q_ab ≥ Q_ij`. Como `Q_ab = (m-2) d_ab - r_a - r_b`, una
cota suficiente barata es
`(m-2)·d_a^{(k)} - r_a - r_max ≥ Q_ij` con `d_a^{(k)}` la k-ésima distancia
menor de `a` y `r_max = max r`. Si la cota falla para algún `a`, se
ejecuta la reducción exacta de NJ0 en esa ronda (`exact_fallbacks++`).
Resultado: **siempre exacto** (I1 se mantiene); la ganancia depende de la
tasa de fallbacks.

Pre-requisito empírico (esta entrega): medir `Recall@k` y la fracción de
rondas certificables sobre los 31 genes reales **con la referencia CPU
instrumentada** (`tools/nj_trace.cpp`), antes de escribir un solo kernel.
Umbral para abrir NJ1: Recall@32 ≥ 0.95 en los genes con n ≥ 1000 y
fallbacks estimados ≤ 10 % de rondas. Si no se cumple, NJ1 se descarta y
NJ0 es la entrega.

**Resultado (job 29238504, `docs/RESULTADO_NJ_TRACE.md`): gate NO
superado.** Recall@32 = 0.90 (CYTB) y 0.93 (COI); la cota de
certificación por filas sólo cierra 1.6 % (CYTB) / 5.8 % (COI) de las
rondas a k=32, es decir ≥ 94 % de fallbacks exactos. NJ1 queda como
propuesta no habilitada; NJ0 es la entrega.

Generadores de candidatos, en orden de coste (el draft §5.2):
(a) top-k sobre `d` actual (recomputado por ronda: Θ(m·k·log) en GPU),
(b) grafo estático de hojas propagado por unión al fusionar (cero
recomputo; lo que un RapidNJ-like haría), (c) sketches / embeddings
hiperbólicos — **fuera de alcance** hasta que (a) o (b) muestren recall
insuficiente.

### NJ2 — merges disjuntos por lote (especulativo)

Fusionar en una misma ronda un matching `M ⊆ E_k` de cerezas mutuamente
disjuntas y certificadas. Rompe la semántica secuencial (I1) salvo que
se demuestre que el orden de merge de las cerezas de `M` no altera el
resultado — en general **no** es cierto (las `r_a` cambian). Sólo se
abre si NJ1 queda limitado por número de rondas y no por trabajo por
ronda. Requiere redefinir la referencia y el gate (RF y likelihood, no
igualdad de `nodes`). No se planifica todavía.

## 3. Lo que este spec NO cubre

- Soft-NJ / gradientes / embeddings hiperbólicos (draft §3–4): son
  herramientas para *búsqueda* de candidatos o para inferencia
  variacional, no para el árbol guía de genomsa. Se retoman, si acaso,
  en NJ1(c).
- Distancias evolutivas corregidas (JC, K2P): genomsa usa Jaccard de
  k-mers. La entrada de NJ es una matriz `float` cualquiera; el kernel
  de distancias es otra pieza (`kmer_distances_mt`, 35.8 s en CYTB, el
  segundo cuello, spec aparte).
- Longitudes de rama: el árbol guía sólo necesita topología y orden.

## 4. Set de gate

- Sintéticos: n ∈ {3..40} aleatorios (ya en `test_msa_ref`), más casos
  con empates exactos construidos (distancias iguales) para ejercitar el
  desempate.
- Reales: los 31 genes de `genes_qc_pass` (n = 140 … 3523), distancias
  producidas por `kmer_distances_mt(k=5)`; el árbol `nj_tree` de cada uno
  se persiste como oráculo (`results/nj_trace/<gen>.tree.tsv`).

## 5. Métricas a reportar (draft §8)

- Exactitud: I1 pass/fail por gen; rondas con empate a ≤1 ulp.
- Rendimiento: `upload_s`, `rounds_s` (desglosado rowsum/argmin/merge),
  `download_s`, `dev_bytes`; host `nj_tree_mt` como línea base.
- NJ1: Recall@k para k ∈ {8,16,32,64,128} (modos (a) y (b)), tamaño
  medio de `E_k`, fracción de rondas certificables.

## 6. Decisiones abiertas

- **D1 — orden de suma de `r_a`.** Opciones: (i) la GPU serializa la
  suma por fila (un hilo por fila: Θ(m) por hilo, m hilos; a m=3.5k son
  3.5k pasos secuenciales — lento pero bit-exacto con la referencia
  actual); (ii) redefinir la referencia con reducción en árbol de bloques
  de 256 y actualizar `nj_tree`/`nj_tree_mt` (rompe bit-exactitud con
  árboles ya publicados; hay que regenerar `results/msa_gpu`); (iii)
  aceptar I1 como único gate y tratar los empates numéricos como riesgo
  medido. Recomendación: empezar con (i) en NJ0 para tener I1 sin
  discusión, medir cuánto cuesta, y decidir (ii) con datos.
- **D2 — dónde vive la matriz entre rondas.** Densa `(2n-1)²` (simple,
  I4 ok a n≤8k) vs `m×m` compactada (RapidNJ) que exige remapear índices
  y complica el desempate. NJ0 usa densa.
- **D3 — `float` vs `double` en device.** La referencia usa `double`;
  MI210 tiene FP64 completo, las NVIDIA de consumo (q6000, pro6000) no.
  NJ0 usa `double`; medir si FP64 limita en NVIDIA.
