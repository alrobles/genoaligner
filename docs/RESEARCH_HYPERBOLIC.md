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

## 4. Experimentos propuestos

1. **Baseline de distorsión** (CPU/Python, horas): k-mer dists del set
   CYTB/COI → Poincaré embedding d=2..8 (geoopt/torch) → stress vs.
   distancias originales; RF del árbol decodificado vs. NJ. Decide si el
   nivel A es viable.
2. **Gate de calidad**: mismo input, genomsa con `--guidetree-in` NJ vs.
   decodificado-hiperbólico → SP/TC en benchmark proteico subset +
   familias sintéticas con verdad conocida (simgrid ya genera
   `sim_true_tree.nwk` — RF directo contra la verdad también).
3. **Scaling**: n=10⁴..10⁵, tiempo end-to-end del path embedding+decode
   vs. NJ actual — objetivo: quitar la barrera de n=10⁴ del simgrid.

## 5. Riesgos y honestidad

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
