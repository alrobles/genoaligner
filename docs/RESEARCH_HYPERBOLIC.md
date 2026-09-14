# Research digest — embeddings hiperbólicos para guide trees y filogenia

Motivación interna: el benchmark sintético identificó el guide tree como
cuello de botella real — k-mer distances O(n²) + NJ O(n³) dominan a
n≈10⁴ (celda n10000/L100: genomsa 1233s vs MAFFT 9.9s, con SPS
equivalente). La literatura de embeddings hiperbólicos ofrece una ruta
GPU-nativa y diferenciable para atacar ese paso, y un ángulo de paper
propio. Refs en `docs/references.bib`.

## 1. Por qué hiperbólico y no euclidiano

Un árbol métrico es 0-hiperbólico (Gromov): embebe en espacio hiperbólico
con distorsión baja acotada, mientras que en espacio euclidiano la
distorsión crece con el número de hojas. Consecuencia práctica: n taxa
caben en H^d con d chica (2–8) y las distancias geodésicas aproximan las
distancias patristicas — lo que para nosotros significa que **un guide
tree se puede construir desde un embedding de dimensión fija en vez de
la matriz O(n²) completa**.

## 2. Digest por referencia

- **Sarkar 2012** (Delaunay embedding): construcción algorítmica exacta —
  dado un árbol, lo embebe en H² con distorsión baja y el grafo de
  Delaunay hiperbólico del embedding contiene las aristas del árbol.
  Dirección inversa que nos interesa: embeber taxa → Delaunay →
  candidato de árbol. Base teórica del "decode" geométrico.
- **Nickel & Kiela 2017** (Poincaré embeddings): aprendizaje de
  embeddings en la bola de Poincaré por Riemannian SGD sobre pares de
  distancia. Es el mecanismo para pasar de nuestras distancias k-mer a
  puntos en H^d sin factorizar la matriz completa: optimización
  estocástica, batches en GPU, costo por iteración independiente de n².
- **Ganea et al. 2018** (HNN): capas neuronales en variedad hiperbólica —
  necesario solo si el embedding lo produce un encoder sobre features
  (k-mers) en vez de fitting directo de coordenadas.
- **Macaulay & Fourment 2024 (Dodonaphy)**: taxa → puntos en espacio
  hiperbólico + **soft-NJ**, un decode diferenciable de NJ (softmin sobre
  los pasos de aglomeración) que permite propagar gradiente de la
  likelihood filogenética al embedding. VI sobre distribuciones de
  embeddings. Limitación honesta reportada por los autores: "geometric
  frustrations" → óptimos locales. Código: github.com/mattapow/dodonaphy
  (incluye soft-NJ reutilizable).
- **Mimori & Hamada 2023 (GeoPhy, NeurIPS)**: VI completamente
  diferenciable sobre todo el espacio de topologías sin restringir
  candidatos — representación geométrica continua de la distribución
  topológica + control variates. Supera a métodos Bayesianos aproximados
  sobre topologías completas.
- **Chen et al. 2025 (H-Vcsmc, AISTATS)**: CSMC/NESTED-CSMC en espacio
  hiperbólico — propuestas secuenciales que construyen el árbol
  partícula a partícula con estimadores insesgados; mejora velocidad y
  escalabilidad vs. versiones euclidianas. Es la variante más cercana a
  "construcción de árbol masiva en GPU".
- **Lewandowski 2009** (vines/onion): generación de matrices de
  correlación aleatorias — en esta literatura se usa para la familia
  variacional sobre correlaciones del embedding; para nosotros sirve
  para priors/inicialización y para simular benchmarks con estructura
  de correlación controlada.
- **Felsenstein 1981 / Semple & Steel / Saitou-Nei NJ**: fundamentos —
  la likelihood que GeoPhy/Dodonaphy optimizan, la teoría de árboles, y
  el baseline que el decode debe igualar o superar.

## 3. Encaje en genomsa (tres niveles, por orden de riesgo)

**Nivel A — guide tree hiperbólico escalable (sustituye NJ a n grande).**
k-mer distances (ya GPU-paralelizable, hoy `_mt`) → embedding Poincaré
por RSGD sobre pares sampleados (O(n·d·iters), sin materializar n²
completo si se muestrea) → decode a binario: hiperbólico-MST/Delaunay o
NJ sobre distancias geodésicas (la matriz geodésica es d-aproximación de
la k-mer, NJ sigue O(n³) pero sobre datos comprimidos — mejor aún:
decode jerárquico por clustering en el disco). Gate de validación:
RF(guía-hiperbólica, guía-NJ) sobre las mismas distancias + SP/TC del
MSA resultante — el guide tree es heurístico, tolera RF>0 si la calidad
de alineamiento se mantiene (igual que UPGMA vs NJ).

**Nivel B — decode diferenciable (paper angle).**
Implementar soft-NJ de Dodonaphy sobre nuestro pipeline: embedding
aprendido minimizando no la discrepancia de distancias sino el costo del
alineamiento progresivo resultante (SP-score proxy o likelihood
filogenética del árbol guía). Contribución propia: guide tree optimizado
para el objetivo que realmente importa (calidad del MSA), no para la
matriz de distancias. Riesgo: óptimos locales documentados + cómputo de
gradiente a través del decode — PyTorch en GPU, escala moderada
primero (n≤1k).

**Nivel C — inferencia filogenética completa (fuera de genomsa-core).**
GeoPhy/H-Vcsmc como backend alternativo a IQ-TREE para la supermatrix —
proyecto separado (phylogenyAI), no mezclar con el milestone de MSA.

## 4. Experimento H1 ejecutado — viabilidad confirmada con límites

`scripts/hyp_embed_test.py` (numpy puro, Poincaré ball + RSGD con
gradiente verificado por diferencias finitas — la disposición
"x/y" citada de NK2017 da **−grad**, corregido en el script).

Setup: `simgrid` cells con `sim_true_tree.nwk` → (a) distancias
patristicas exactas, (b) distancias k-mer Jaccard k=5 de `sim_in.fasta`
(la métrica real de genomsa) → embedding → NJ sobre geodésicas → RF.

**Resultados:**

| input | dim | rel-stress | RF(NJ(G), verdad) | baseline NJ(D) |
|---|---|---|---|---|
| patristica (s=0.02) | 4 | 0.124 | 43 | 1 |
| patristica (s=0.02) | 8 | 0.054 | **1** | 1 |
| k-mer (s=0.02) | 8 | 0.091 | 21 | 3 |
| k-mer (s=0.02) | 16 | 0.041 | **3** (RF=0 vs NJ) | 3 |
| k-mer (s=0.10) | 16 | 0.074 | 121 | 47 |
| k-mer (s=0.10) | 32 | 0.032 | **43** | 47 |

**Lectura:** (1) la recuperación es esencialmente exacta cuando el
stress baja de ~0.05 — existe un umbral de dimensión que crece con el
ruido de la métrica (16 para s=0.02, ~32 para s=0.10); (2) en el
régimen difícil el embedding incluso suaviza el ruido (43 < 47);
(3) el decode usado sigue siendo NJ O(n³) — el embedding comprime la
información topológica a n·d flotantes pero NO quita el costo de
decode. La escalabilidad real requiere reemplazar el decode (MST/
single-linkage/soft-NJ en H^d) — ese es el siguiente experimento.

## 5a. Experimento H2 ejecutado — embedding como generador de candidatos: NEGATIVO

`scripts/hyp_nj_cand.py` mide, replicando el esquema "static" de
`nj_trace`, si los vecindarios top-k en espacio Poincaré (geodésicas G
tras un embedding de la D k-mer inicial) recuperan la cereza NJ por
ronda mejor que los vecindarios top-k de la propia D.

| datos | cand | r@8 | r@16 | r@32 | r@64 | r@128 |
|---|---|---|---|---|---|---|
| sim n=96 s=0.06 | rawD / hypG d=16 | 1.00 / 0.90 | 1.00 / 0.95 | 1.00 / 0.97 | 1.00 / 0.98 | 1.00 / 1.00 |
| sim n=128 s=0.15 | rawD / hypG d=16 | 0.69 / 0.35 | 0.84 / 0.53 | 1.00 / 0.75 | 1.00 / 0.83 | 1.00 / 1.00 |
| BDNF real n=457 | rawD / hypG d=16 | 0.97 / 0.67 | 0.99 / 0.74 | 1.00 / 0.79 | 1.00 / 0.84 | 1.00 / 0.90 |
| BDNF real n=457 | rawD / hypG d=32 | 0.97 / 0.80 | 0.99 / 0.84 | 1.00 / 0.88 | 1.00 / 0.90 | 1.00 / 0.91 |
| BDNF real n=457 | rawD / hypG d=64 | 0.97 / 0.75 | 0.99 / 0.83 | 1.00 / 0.87 | 1.00 / 0.89 | 1.00 / 0.92 |

**Veredicto: negativo y consistente.** La proyección a H^d preserva la
jerarquía global (stress ≤0.09 a d=32) pero degrada el orden local que
las cerezas necesitan — a d=64 sigue 8 puntos bajo rawD en r@128.
Además rawD ya da r@32≈1.0 en estos datos: el cuello de candidatos que
`nj_trace` midió en genes reales (~90% fresh@32) no lo resuelve el
embedding, lo empeora. Cierra la puerta a "embedding → sparse NJ" y
refuerza el veredicto de §5b: el nicho hiperbólico viable es
**refinamiento** (soft-NJ / verosimilitud sobre embedding), no
generación de candidatos ni sustituto del NJ0-GPU exacto.

## 5. Experimentos pendientes

1. **Decode barato**: single-linkage/MST sobre geodésicas H^d (O(n²) con
   heaps, o Delaunay hiperbólico tipo Sarkar) → comparar RF y tiempo vs.
   NJ completo. Si single-linkage basta para guide tree, el pipeline
   completo queda O(n²d) efectivo.
2. **Gate de calidad MSA**: `--guidetree-in` con el árbol decodificado
   vs. NJ estándar → SP/TC en benchmark proteico subset.
3. **Scaling**: n=10⁴..10⁵ con pares muestreados (sin materializar n²
   distancias) — objetivo: quitar la barrera n=10⁴ del simgrid.

## 5b. Contexto del trabajo paralelo: NJ0-GPU (RESUMEN_NJ_GPU_HALLAZGOS)

La sesión hermana entregó `nj_tree_gpu` — NJ exacto denso en device,
bit-exact con `nj_tree` host, validado MI210/A100/V100 sobre los 31
genes reales (CYTB n=3523: 60s host → ~1.5s device). Esto **reubica el
rol del camino hiperbólico**:

- El cuello O(n³) de NJ ya está cubierto por un kernel exacto —
  un embedding + decode aproximado no compite en esa fila.
- Su medición clave (`nj_trace`, 19k rondas reales): la cereza NJ está
  en el top-32 por distancia de sus extremos sólo 90% de las rondas —
  "vecino cercano ≠ cereza". Un embedding como generador de candidatos
  debe demostrar Recall@32>0.95 antes de merecer kernels.
- Veredicto coincidente con nuestro H1: lo hiperbólico es capa de
  **refinamiento** (NJ0 → embedding → gradiente sobre verosimilitud,
  soft-NJ) y de escalado *sin distancias* (pares muestreados), no
  sustituto del NJ exacto.
- Su plan de follow-up que comparte infraestructura con H1: extender
  `nj_trace` con candidatos por embedding (Sarkar sobre NJ parcial,
  o Poincaré sobre d) medido en Recall@32 — `hyp_embed_test.py` ya
  provee la pieza de embedding.

## 6. Riesgos y honestidad

- El guide tree NO es el árbol filogenético final — sobre-invertir en
  su "calidad" tiene rendimiento decreciente: solo importa el orden de
  merge. La métrica correcta es SP/TC resultante, no RF.
- Dodonaphy reporta local optima issues; nuestro uso (guía para
  progresivo) es más tolerante que inferencia ML exacta.
- n≤3.5k (spec original): NJ O(n³) son segundos — el path hiperbólico
  solo se justifica a n≥10⁴ o por el ángulo diferenciable. No vender
  speedup donde no hace falta.
- Aritmética hiperbólica (exp/log maps, acosh) en float32 es inestable
  cerca del borde del disco — usar float64 o Lorentz model si se
  implementa en kernel; mantener el gate de paridad cross-vendor.
