# MSA Research Digest — base para genomsa

Consolidación de la investigación (subagentes, 2026-09-13). Fuentes citadas
por sección.

## 1. Estado del arte GPU MSA: TWILIGHT (Tseng, Walia, Turakhia 2025)

Bioinformatics 41(S1):i332, github.com/TurakhiaLab/TWILIGHT (MIT).

- **Pipeline**: FASTA + guide tree (Newick) → divide guide tree en
  subtrees ≤m hojas (centroid decomposition, para RAM no accuracy) →
  progressive alignment por subtree → merge (transitivity PASTA-style o
  progressive). Modo add-to-backbone y merge-MSAs existen.
- **Core**: NW global + Gotoh affine gaps, **profile-vs-profile** donde
  perfil = matriz de frecuencias por columna ({A,C,G,T,N,gap},
  estilo ClustalW). Column score = **sum-of-pairs sobre frecuencias**
  `Σ_a Σ_b f_i(a)·g_j(b)·S(a,b)`. Gap penalties **position-specific**:
  gap-open reducido donde el perfil opuesto ya tiene gaps.
- **Scheduling**: post-order traversal; nodos del mismo "order"
  (1+max(children)) son DPs independientes → paralelismo inter-alineamiento
  (1 block GPU por DP) × intra (wavefront anti-diagonal).
- **Defaults nt**: match 18, mismatch -8, transition -4, gap-open -50,
  gap-extend -5.
- **Banding**: X-drop adaptativo (default 600×gap-ext), memoria ~lineal.
  Traceback: **TALCO** (Walia et al., HPCA 2024) — tiles + convergencia de
  punteros → memoria de traceback **constante** (cabe en shared).
- **Gappy-column elision**: quita columnas >95% gap antes de cada DP,
  las reinserta después mergeándolas si coinciden — 12× speedup end-to-end
  en árboles profundos y MSAs 14× más chicos que PASTA/MAGUS merge.
- **GPU**: DPX (Hopper), 2 traceback pointers/byte en shared, HIP port
  para AMD. 10k seqs en 257s (MI210) / 177s (MI300X); 1M RNASim en ~30min.
- **Limitaciones**: depende del guide tree; proteína es nuevo (v0.2);
  X-drop puede ser subóptimo en muy divergentes; nucleotide-centric.
- CPU hace perfiles y gap penalties; GPU solo hace el DP.

## 2. Diseño del espacio de algoritmos (survey)

Veredictos para nuestro spec (150–3,500 seqs × ≤6kb, >70% id, CDS):

- **Progressive profile-profile ES la elección correcta** a esta escala.
  Decomposition (PASTA/MAGUS), regressive, eHMM (UPP/WITCH), DAG-align,
  learnMSA (protein-only) — todos apuntan a N≫10⁴ o a proteína.
- **Guide tree**: k-mer Jaccard corregido para fragmentos
  `D = 1 − S_ij/min(S_ii,S_jj)` (Katoh & Toh 2007) → **NJ** (UPGMA
  aceptable, SLINK NO — árboles caterpillar serializan). A N=3500, NJ
  O(N³) ≈ segundos. Exponer `--guidetree-in` (phylogenyAI ya tiene
  árboles de genes + scores bait del QC).
- **Profile DP**: Gotoh 3-state, **semiglobal (terminal gaps gratis)** —
  los fragmentos reales (CYTB 207bp vs 1137bp) lo exigen; force-global
  = el fallo MAFFT documentado en H14. Gap-open escalado por
  **occupancy** de la columna opuesta (MUSCLE/FAMSA). NO usar WFA para el
  perfil (su wavefront explota costo unitario; no generaliza a vectores
  de frecuencia).
- **IUPAC**: CYTB tiene ~9% ambiguos (N,R,Y…). Perfil = cuentas
  **fraccionarias** (1/|set| por base compatible) — casi gratis en la
  formulación de frecuencias.
- **Codon-aware**: no hace falta en v1 — set pre-filtrado clean-CDS +
  audit post-hoc (indels mod 3, stops). Fallback = DP en espacio aminoácido
  + back-translate (equivalente a lo que MACSE aporta).
- **Clustal Omega** usa profile-HMM-vs-HMM (HHalign, 3-state pair-HMM) —
  más preciso en proteína, doble complejidad de recurrencia; skip.
- **Gap-gap columns**: nunca puntúan como match (el gap symbol queda
  fuera del dot product).
- **Segundo pase FFT-NS-2-style** (rebuild tree desde MSA #1, realinear)
  = flag opcional, ~2× costo, ganancia real documentada (Katoh 2013).
- **Trampas rankeadas**: guide-tree quality > todo; gap-semantics en
  perfiles (gap-gap scoring, occupancy scaling, gap attraction con N
  grande — FAMSA set-size correction); fragments (distancias corregidas +
  terminal gaps); IUPAC; determinismo (tie-breaking en NJ, orden de
  hijos, traceback, batch order); banding overflow (log/re-widen, nunca
  silenciar); pesos por secuencia (clados densos dominan perfil —
  diferible); merge off-by-ones (definir interleave en CPU ref primero).

## 3. Lo que Upham et al. 2019 hizo realmente (subagente, fuentes verificadas)

Pipeline documentado en el main text de PLOS Biol 17(12):e3000494:

**BLAST-bait gathering en NCBI** (209,294 hits → 22,504 seqs) → **loop
iterativo por gen**: (1) alineamiento **guiado por traducción al marco
aminoácido correcto** de los 26 fragmentos codificantes — *el alineador
NO se nombra en el texto* (plausiblemente Geneious v9.1 translation-align,
documentado para concatenación); (2) error-checking: stops + non-overlap
completo → excluir; (3) árbol de gen **RAxML v8.2.3** con outgroups
designados (Anolis/Monodelphis/Ornithorhynchus) → remover rogue taxa →
concatenación en **Geneious v9.1** (39,099 bp × 4,098 spp, 11.9% completo,
21,021 seqs) → **PartitionFinder v1.1.1: 9 particiones** (APP+CREM+FBN1,
BMI1, PLCB4, nDNA pos-codón 1/2/3, mtDNA pos-codón 1/2/3) bajo GTR+G.

Confirmaciones para nuestro trabajo:

- **APP, BMI1, CREM, FBN1, PLCB4 son loci "NC" (noncoding) en su Tabla 2**
  — confirma independientemente nuestro hallazgo de que no hay record
  same-locus que traduzca limpio (los amplicones contienen intrones/UTR).
  En su esquema de partición van con los no-codificantes, no por codón.
- Su exclusion total: 1,618 seqs (7.2%) vs nuestro QC+filter: 180
  drops explícitos + la curación previa por longitud.
- El alineador sin nombrar y el loop align→check→tree es exactamente el
  hueco que un genomsa + gene_qc llenan de forma auditable.
- Trimming: NO documentado (ningún tool nombrado).
- Detalle fino (umbral de stops, rogue-detection, trimming) vive en
  S1 Text §3 — requiere descarga directa del suplemento PLOS/Dryad
  (doi:10.5061/dryad.tb03d03); TODO si se necesita exactitud.
- PartitionFinder→9 particiones: nuestro supermatrix actual usa
  partición-por-gen; el esquema Upham agrupa por posición de codón —
  decidir cuál reportar (ambos defendibles, documentar diferencia).

## 4. Decisiones fijadas para M1 (referencia CPU)

| componente | elección |
|---|---|
| distancia | k-mer k=5 nt, Jaccard corregida `1−S/min(Sii,Sjj)` |
| árbol | NJ (determinista, tie-break por índice); `--guidetree-in` override |
| perfil | frecuencias {A,C,G,T} + fraccionario IUPAC + occupancy float |
| DP | Gotoh 3-state, semiglobal (terminal gaps libres), unbanded |
| column score | Σ_a Σ_b f·g·S(a,b); gap no participa |
| gap-open | escalado por occupancy opuesta |
| batch | nodos mismo depth → batch API existente |
| merge | interleave determinístico por CIGAR; hijos ordenados por árbol |
| validación | SP/TC vs MAFFT/MACSE; RF de IQ-TREEs; audit codón |
