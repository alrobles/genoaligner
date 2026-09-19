# PLAN — GPU MSA ("genomsa")

Sprint para construir un alineador múltiple GPU dentro de genoaligner,
usando la misma disciplina: paridad primero, oráculos externos, shim CPU,
evidencia honesta. El caso de uso driver: alinear los 31 genes del
pipeline phylogenyAI (150–3,500 seqs × 300–6,000 bp CDS, >70% similar).

## Por qué es viable en sprint (y dónde NO aspiramos)

- Nuestra escala es ~3 órdenes menor que el diseño de TWILIGHT (millones
  de seqs, árboles gigantes). A 3.5k seqs × 6kb NO hace falta
  divide-and-conquer ni PASTA merge: el alineamiento progresivo puro son
  n-1 DPs perfil-vs-perfil, cada uno del mismo tamaño que los kernels
  pairwise que ya tenemos verificados.
- El kernel perfil-vs-perfil es la MISMA forma de DP que pairwise
  (rejilla M×N, dependencias anti-diagonales) con scoring de columna vs
  columna (dot product de frecuencias) en vez de match/mismatch escalar.
- ClipKIT-class trimming es trivial — fuera de alcance.

## Fases

### Fase M0 — Investigación y spec (esta sesión)
- Métodos Upham exactos, arquitectura TWILIGHT, survey de diseño
  (subagentes en curso → consolidar en docs/MSA_RESEARCH.md).

### Fase M1 — Referencia CPU correcta
- `msa_ref` CPU: árbol guía (kmer Jaccard + NJ/UPGMA), post-order
  traversal, NW global perfil-vs-perfil sum-of-pairs, gaps afines.
- Semántica primero: definir cómo se representan gaps en perfiles
  (frecuencia de '-' por columna), inserción de columnas gap-gap,
  orden de hijos.
- Tests sintéticos: 3-10 seqs con alineamiento conocido; paridad
  column-vs-column vs MAFFT en genes chicos (ADRB2 n=163).

### Fase M2 — Kernel GPU perfil-vs-perfil
- Kernel `msa_profile_kernel`: warp-per-row como sw_kernel, scoring
  columna×columna (5 estados ACGT-), batch de alineamientos de perfiles
  (los nodos del mismo nivel del árbol son independientes → batch
  natural del API existente).
- Traceback → fusión de perfiles (operación nueva: intercalar columnas
  gap según el traceback de cada lado).
- Gate: scores GPU == CPU ref en cada nodo; CIGARs rescore-ok.

### Fase M3 — Escala y robustez
- Batch por niveles del árbol (todos los nodos hoja-adyacentes en
  paralelo, luego el siguiente nivel).
- Auto-shrink si un DP excede workspace (maquinaria ya existe).
- Banding X-drop posterior — SOLO si la evidencia lo pide; a 6k×6k el
  DP completo cabe en memoria (72MB), banda es optimización no
  requisito.

### Fase M4 — Validación vs referencias
- SP/TC score vs MAFFT --auto y MACSE en los 31 genes.
- Comparación funcional: árbol IQ-TREE de cada MSA → RF distance
  entre topologías.
- Codon check: stops internos post-alineamiento (la garantía
  codon-aware la da el filtrado previo + auditoría, no el alineador).

### Fase M5 — Integración pipeline
- `tools/geno_msa.cpp` CLI (fasta in → aln fasta out) + sbatch.
- Re-correr los 31 genes, comparar, documentar RESULTADO_MSA.

## Riesgos principales (honestos)
1. Guide-tree quality domina la calidad final — un mal árbol guía
   produce MSA malo aunque el DP sea perfecto. Mitigar: NJ sobre
   distancias reales (tenemos scores vs bait del QC ya!) o usar el
   árbol de genes que el pipeline ya computa.
2. Perfiles con gaps: la semántica de frecuencias de '-' puede
   "desalinear" regiones gappy (el problema que TWILIGHT resuelve con
   su heurística). M1 la define explícitamente.
3. Orden de hijos / determinismo: definir y testear.
4. Un MSA que "se ve bien" pero es peor que MAFFT es el peor outcome —
   el gate de M4 es real, no cosmético.
