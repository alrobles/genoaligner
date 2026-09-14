# RESULTADO — traza NJ: ¿está la cereza exacta en un vecindario top-k?

Gate empírico de la fase NJ1 de `docs/SPEC_NJ_GPU.md`. Antes de escribir
un kernel sparse-candidate se midió, con la referencia CPU exacta
instrumentada (`tools/nj_trace.cpp`), en qué fracción de las rondas de NJ
la cereza `(i,j) = argmin Q` cae dentro de los k vecinos más cercanos por
distancia de alguno de sus extremos, y en cuántas rondas la cota de
certificación por filas habría permitido aceptar la propuesta sin la
reducción densa.

- Job KU HPC **29238504** (`sixhour`, 16 CPU, r10r10n01), commit `60ea7b3`,
  clon `/beegfs/a474r867/genoaligner/nj_trace_wt`.
- Datos: 31 genes de `/beegfs/a474r867/phylogenyAI/data/genes_qc_pass/`
  (n = 140 … 3523). Distancias k-mer (k=5) idénticas al pipeline MSA.
- Tiempo total del job: 2 min 46 s (CYTB: 97 s de traza; el NJ exacto
  `nj_tree_mt` con 16 hilos tarda 11.8 s).
- La traza reconstruye `Tree.nodes` y se comprueba **idéntica** a
  `nj_tree_mt` en los 31 genes (el binario sale con rc=3 si no; ningún
  gen falló).
- Tabla completa: `docs/nj_trace_summary_29238504.tsv`; trazas por ronda
  en `$PAI/data/nj_trace/rounds/<gen>.tsv`.

## Métricas

Por ronda (m nodos vivos, cereza `(i,j)`):

- `fresh@k`: `min(pos(j en topk(i)), pos(i en topk(j))) < k` con el top-k
  calculado sobre la `d` actual (generador (a) del spec).
- `static@k`: igual, pero con el top-k de cada **hoja** calculado una vez
  sobre la `D` inicial y los clusters heredando la unión de los conjuntos
  de sus hojas (generador (b), RapidNJ-like, cero recomputo).
- `cert@k`: fracción de rondas en las que **todas** las filas cumplen
  `(m-2)·d_a^{(k)} − r_a − r_max ≥ Q_best` (cota suficiente barata del
  spec §NJ1). Rondas no certificadas = fallback exacto obligatorio.
- `ties`: rondas con segundo mejor `Q` a ≤ 4 ulp del mejor (riesgo de
  desempate en un argmin paralelo).

## Resultados (ordenado por n)

| gene | n | nj_s | trace_s | ties | fresh@8 | fresh@32 | fresh@128 | static@32 | static@128 | cert@32 | cert@128 | max_rank_fresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CYTB | 3523 | 11.84 | 96.77 | 2 | 0.8625 | 0.9009 | 0.9296 | 0.9565 | 0.9974 | 0.0156 | 0.0386 | 128 |
| COI | 1608 | 1.54 | 9.68 | 2 | 0.8400 | 0.9303 | 0.9664 | 0.8400 | 0.9172 | 0.0579 | 0.1270 | 128 |
| IRBP | 1252 | 0.97 | 4.06 | 3 | 0.9904 | 0.9984 | 1.0000 | 0.9944 | 0.9992 | 0.2944 | 0.6320 | 36 |
| ND1 | 941 | 0.67 | 1.89 | 2 | 0.8253 | 0.8605 | 0.9318 | 0.9883 | 1.0000 | 0.2758 | 0.4952 | 128 |
| ND2 | 940 | 0.66 | 1.89 | 2 | 0.8454 | 0.8795 | 0.9446 | 0.9584 | 0.9989 | 0.2409 | 0.4414 | 128 |
| GHR | 926 | 0.65 | 1.84 | 2 | 0.9113 | 0.9935 | 1.0000 | 0.9913 | 1.0000 | 0.2273 | 1.0000 | 51 |
| BRCA1 | 913 | 0.67 | 1.81 | 5 | 0.6696 | 0.9780 | 0.9989 | 0.9177 | 0.9945 | 0.0406 | 0.1416 | 128 |
| RAG1B | 855 | 0.58 | 1.50 | 2 | 0.8746 | 0.9742 | 1.0000 | 0.9343 | 1.0000 | 0.0574 | 0.4349 | 59 |
| RAG2 | 836 | 0.57 | 1.44 | 3 | 0.9676 | 0.9892 | 1.0000 | 0.9736 | 0.9916 | 0.0707 | 0.1787 | 106 |
| VWF | 755 | 0.50 | 1.12 | 2 | 0.3811 | 0.9934 | 1.0000 | 0.9987 | 1.0000 | 0.2457 | 1.0000 | 44 |
| RAG1A | 553 | 0.34 | 0.53 | 2 | 0.8149 | 0.9510 | 0.9964 | 0.8748 | 0.9891 | 0.1016 | 0.4846 | 128 |
| APOB | 509 | 0.31 | 0.45 | 2 | 0.9112 | 0.9822 | 1.0000 | 0.9507 | 1.0000 | 0.0809 | 0.6331 | 90 |
| BDNF | 457 | 0.27 | 0.36 | 4 | 1.0000 | 1.0000 | 1.0000 | 0.9956 | 1.0000 | 0.2088 | 1.0000 | 7 |
| PLCB4 | 430 | 0.25 | 0.31 | 3 | 0.9907 | 1.0000 | 1.0000 | 0.9977 | 1.0000 | 0.2079 | 0.9276 | 28 |
| ADORA3 | 384 | 0.25 | 0.25 | 3 | 0.9686 | 1.0000 | 1.0000 | 0.9895 | 0.9974 | 0.2435 | 1.0000 | 28 |
| ATP7 | 379 | 0.23 | 0.24 | 2 | 0.9735 | 0.9973 | 1.0000 | 0.9867 | 1.0000 | 0.0875 | 0.8355 | 32 |
| PNOC | 365 | 0.21 | 0.22 | 2 | 0.9862 | 1.0000 | 1.0000 | 0.9945 | 0.9972 | 0.3306 | 1.0000 | 25 |
| APP | 359 | 0.22 | 0.22 | 6 | 0.9664 | 0.9972 | 1.0000 | 0.9916 | 1.0000 | 0.1653 | 0.7871 | 47 |
| DMP1 | 352 | 0.20 | 0.21 | 2 | 0.9600 | 0.9943 | 1.0000 | 0.9800 | 1.0000 | 0.4029 | 1.0000 | 38 |
| CNR1 | 306 | 0.19 | 0.16 | 3 | 0.9704 | 1.0000 | 1.0000 | 0.9934 | 1.0000 | 0.2961 | 0.8882 | 27 |
| TYR1 | 289 | 0.16 | 0.14 | 3 | 0.7944 | 0.9965 | 1.0000 | 0.9930 | 1.0000 | 0.1394 | 1.0000 | 32 |
| CREM | 283 | 0.13 | 0.15 | 3 | 0.9786 | 1.0000 | 1.0000 | 0.9858 | 1.0000 | 0.3630 | 1.0000 | 25 |
| TTN | 269 | 0.15 | 0.12 | 2 | 0.8315 | 0.9176 | 1.0000 | 0.8876 | 1.0000 | 0.1124 | 0.9288 | 112 |
| EDG1 | 255 | 0.14 | 0.11 | 3 | 0.9763 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5217 | 1.0000 | 16 |
| FBN1 | 252 | 0.14 | 0.11 | 3 | 0.9920 | 1.0000 | 1.0000 | 0.9880 | 1.0000 | 0.2960 | 0.5840 | 24 |
| BCHE | 242 | 0.15 | 0.11 | 4 | 0.9667 | 1.0000 | 1.0000 | 0.9833 | 0.9958 | 0.2792 | 1.0000 | 22 |
| A2AB | 238 | 0.13 | 0.10 | 2 | 0.9788 | 1.0000 | 1.0000 | 0.9958 | 1.0000 | 0.7542 | 1.0000 | 22 |
| BRCA2 | 229 | 0.13 | 0.09 | 3 | 0.9031 | 0.9692 | 1.0000 | 0.9295 | 1.0000 | 0.1366 | 0.8767 | 46 |
| ADRB2 | 163 | 0.09 | 0.06 | 2 | 0.9627 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8758 | 1.0000 | 19 |
| ENAM | 153 | 0.08 | 0.05 | 2 | 0.8874 | 1.0000 | 1.0000 | 0.9801 | 1.0000 | 0.6755 | 1.0000 | 29 |
| BMI1 | 140 | 0.08 | 0.05 | 2 | 0.9928 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2464 | 0.9130 | 9 |

## Lectura

1. **Recall en los genes grandes (n ≥ 1000) — gate no superado.**
   CYTB (n=3523): fresh@32 = 0.90, fresh@128 = 0.93; COI (n=1608):
   0.93 / 0.97. El umbral era ≥ 0.95 a k=32. Sólo IRBP (n=1252) lo cumple
   (0.998). El `max_rank_fresh` = 128 en CYTB, COI, ND1, ND2, BRCA1,
   RAG1A indica que hay rondas en las que la cereza no está ni entre los
   128 vecinos más cercanos de ninguno de sus extremos: la cereza NJ no es
   un par "cercano" en `d`, es el par que minimiza `Q`, y `r_i + r_j`
   domina en las rondas tardías (clusters grandes con `d` promediada).
2. **El grafo estático (b) no es peor que el fresco (a)** — de hecho es
   mejor en CYTB (static@32 = 0.957, static@128 = 0.997) y en los ND*.
   Esto es lo que explota RapidNJ. Pero sigue lejos de 1.0 y no da ninguna
   garantía por ronda.
3. **La certificación barata es inútil.** cert@32 = 1.6 % (CYTB), 5.8 %
   (COI), mediana 24 % sobre los 31 genes. La cota
   `(m-2)·d_a^{(k)} − r_a − r_max` es demasiado holgada porque `r_max`
   puede ser mucho mayor que el `r_b` de los candidatos reales. Con
   ≥ 94 % de fallbacks exactos en los genes que importan, NJ1 costaría más
   que NJ0 (candidatos + certificación + reducción densa casi siempre).
   Una cota más fina (p. ej. `r_max` restringido a los `b ∉ E_k(a)`, o
   usar el `r_b` real de cada candidato) es posible pero ya no es "barata"
   y sigue sin garantizar la tasa de fallbacks.
4. **Empates de `Q`**: 83 de 19 094 rondas (0.43 %) tienen dos pares a
   ≤ 4 ulp. Cualquier argmin paralelo (GPU) **debe** reproducir el
   desempate "primer par en orden `(a<b)` sobre `alive`" del host o
   fallará I1 en ~0.4 % de las rondas, lo que basta para cambiar la
   topología. Esto fija la decisión D1 del spec: reducción por bloques con
   comparador `(q, a, b)` lexicográfico, no `atomicMin` sobre `q`.
5. **Escala**: `nj_tree_mt` con 16 hilos ya baja CYTB a 11.8 s (el 60 s
   de RESULTADO_MSA_GPU2 salió de un job con `--cpus-per-task=8`;
   probablemente hilos + nodo distinto). Esto acota la
   ganancia de NJ0: el objetivo razonable es ≤ 2 s en MI210 para
   n=3523, no "de minutos a segundos".

## Decisión

- **NJ1 descartada** como línea de entrega (spec §NJ1 actualizado).
  Queda el binario `nj_trace` para re-evaluar si cambia la métrica de
  distancia o si aparece un generador de candidatos con recall ≈ 1
  (embeddings), pero no se escribirán kernels sparse.
- **NJ0 exacto denso en GPU es la entrega**: rowsum + argmin
  lexicográfico + merge en device, `alive` en host, paridad bit-exact
  con `nj_tree` sobre los 31 genes como test de aceptación.
- Contribución secundaria: los empates reales (0.4 % de rondas) obligan a
  que NJ0 use FP64 con el **mismo orden de suma** que el host en `r`, o a
  demostrar con este tracer que la suma en otro orden no cambia ningún
  argmin en el set de gate (D1).

## Reproducir

```bash
sbatch --export=ALL,REPO=/beegfs/a474r867/genoaligner/nj_trace_wt scripts/nj_trace.sbatch
# local, un gen:
g++ -O2 -std=c++17 -Iinclude -o nj_trace tools/nj_trace.cpp src/msa/msa_ref.cpp -lpthread
./nj_trace gene.fasta --name GENE --header --rounds rounds.tsv
```
