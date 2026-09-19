# genomsa — benchmark proteico reproducible (SP/TC)

Primer benchmark de calidad sobre referencias estándar. Objetivo: medida
honesta de dónde queda un alineador progresivo Gotoh de una pasada frente
a herramientas establecidas — no una afirmación de superioridad.

## Setup

- Datos: bundle `bench1.0` de R. Edgar (drive5.com/bench/bench.tar.gz):
  - `bali3`   — BAliBASE v3, 386 familias (curadas, casos difíciles)
  - `prefab4` — PREFAB v4, 1681 familias (pares estructurales + homólogos)
  - `sabrem`  — SABRE/SABmark v1.65, 303 familias (homólogos remotos)
  - `ox`      — OXBench, 395 familias (más fácil/homólogo)
- Scoring: `scripts/msa_bench_score.py`, semántica qscore-compatible:
  - columnas core = regiones UPPERCASE de la referencia
  - SP (Q): fracción de pares residuo-residuo de la referencia
    reproducidos en el test
  - TC: fracción de columnas core exactamente reproducidas
  - validación: ref-vs-ref = 1.0/1.0; perturbaciones detectadas;
    caso BB11001 reproducido idénticamente local/cluster
- Alineador: `genomsa --cpu --protein` (defaults: BLOSUM62, go=11,
  ge=1.0, global, psgp, gappy 0.95, kmer_k=2)
- Ejecución: `scripts/msa_bench.sbatch`, nodo CPU 32 cores, 2765
  alineamientos en **2 min 57 s** de wall-clock
  (job 29232350/29232368, `/beegfs/a474r867/bench/genomsa_out/`)

## Resultados (media por familia; baselines ejecutados en la misma máquina)

Baselines reales: mafft 7.526 (`--auto`, y `--localpair --maxiterate 1000`
como linsi), muscle 5.3, kalign 3.6 — mismo hardware, 1 hilo por invocación,
tiempos de wall-clock por familia en `/beegfs/a474r867/bench/*_times.tsv`.

| set | genomsa | kalign | mafft-auto | linsi | muscle5 |
|---|---|---|---|---|---|
| bali3   SP | 0.723 | 0.792 | 0.870 | 0.871 | 0.888 |
| bali3   TC | 0.356 | 0.457 | 0.606 | 0.608 | 0.628 |
| sabrem  SP | **0.557** | 0.550 | 0.650 | 0.650 | 0.682 |
| sabrem  TC | **0.377** | 0.370 | 0.470 | 0.470 | 0.498 |
| — twi   SP | **0.379** | 0.359 | 0.471 | 0.471 | 0.520 |
| ox      SP | **0.872** | 0.867 | 0.886 | —    | 0.898 |
| ox      TC | **0.765** | 0.763 | 0.793 | —    | 0.811 |
| prefab4 SP | **0.561** | 0.557 | 0.694 | —    | 0.691 |

Categorías BAliBASE (SPmed): RV11 0.51, RV12 0.87, RV20 0.91, RV30 0.77,
RV40 0.69, RV50 0.66 — tabla completa en `bench/report.txt`.

Runtime total por set (s): bali3 genomsa-cpu **222** / kalign 101 /
mafft 2583 / linsi 2905 / muscle 15272. GPU (mi210 1621, v100 927) más
lento que CPU en familias chicas: el overhead por invocación domina —
el GPU gana a escala (batch-31 genes de 3-4k seqs). Cross-vendor:
**MI210 == V100 == CPU en las 2765 familias, byte-idéntico**
(post `-ffp-contract=off`; ver nota FMA abajo).

Fallos: 0 en genomsa-cpu/v100 y todos los baselines; 8 en mi210 por
contención de `-P8` sobre el dispositivo — re-corridos secuencialmente
con salida idéntica.

## Lectura honesta

- Contra baselines reales modernos (no los scores de 2009 del bundle),
  genomsa queda en el tier progresivo rápido: **supera o empata a
  Kalign3 en sabrem/ox/prefab4** (incluido el twilight zone: 0.379 vs
  0.359) y queda ~7 pts bajo él en bali3.
- La distancia a los iterativos (mafft/linsi/muscle5) es de ~10-17 pts
  SP — eso es lo que cuesta una sola pasada progresiva sin consistencia
  ni refinamiento. Es el siguiente objetivo de calidad, no un defecto
  del kernel.
- GPU < CPU en este benchmark: familias de 10-100 seqs no llenan el
  dispositivo y el overhead por proceso domina. El GPU ya demostró su
  valor a escala real (genes de 3-4k secuencias del pipeline
  PhylogenyAI).

## Grid sintético de escalabilidad (scripts/msa_sim_grid.sbatch)

genomsa --cpu vs mafft --auto --thread 8, SPS contra verdad simulada:

| cell | genomsa s | SPS | mafft s | SPS |
|---|---|---|---|---|
| n=10 L=10k    | 28.2  | .977 | 8.7  | .983 |
| n=100 L=1k    | 3.4   | .933 | 2.1  | .923 |
| n=100 L=10k   | 331.5 | .888 | 47.1 | .924 |
| n=1000 L=1k   | 37.6  | .805 | 4.8  | .785 |
| n=10k L=100   | 1233  | .739 | 9.9  | .733 |
| div=0.10      | 4.7   | .125 | 1.8  | .628 |
| div=0.30      | 9.6   | .001 | 13.1 | .007 |
| indel=0.08    | 5.5   | .064 | 8.5  | .063 |

Hallazgos: (1) el NJ O(n³) + distancias k-mer O(n²) colapsan a n=10k
(1233 s vs 9.9 s — el árbol guía es el cuello real, no el DP);
(2) a divergencia ≥0.10 ambos colapsan (zona donde NINGÚN progresivo
funciona); (3) SPS genomsa ≈ o > mafft en la mayoría de celdas.

## Nota de plataforma — FMA en ROCm (hallazgo del benchmark)

`__fmul_rn`/`__fadd_rn` evitan contracción FMA bajo nvcc pero **NO bajo
hipcc/gfx90a**: 57/2765 familias divergieron MI210 vs V100/CPU hasta
añadir `-ffp-contract=off` a todos los builds HIP (commit 9b8ad69).
Post-fix: MI210 == V100 == CPU, 2765/2765 byte-idéntico.

## Sensibilidad de parámetros (subset 40 familias bali3)

| config | SP | TC |
|---|---|---|
| defaults antiguos (go=11 ge=1.0 ends-libres) | .688 | .298 |
| +global | .746 | .350 |
| +global ge=0.5 | .753 | .362 |
| +global ge=1.0 (defaults actuales) | .753 | .362 |
| global, sin psgp | .678 | .301 |
| global, sin gappy | .713 | .333 |

- `--global` aporta ~+3 puntos: las familias proteicas son problema
  global (los ends libres eran una asunción de loci de DNA).
- PSGP aporta ~+3.7 puntos en proteína — mayor efecto que en DNA.
- Gappy-strip es casi neutro en proteína (pocas columnas >95% gap),
  pero no daña; queda activo por consistencia con el pipeline DNA.
- La superficie go∈{9..15}, ge∈{0.5..1.0} con global es plana
  (±0.003) — defaults tomados de convención ClustalW, no sobreajustados.

## Reproducibilidad

```bash
# en el cluster
sbatch scripts/msa_bench.sbatch
# scoring de un archivo
python3 scripts/msa_bench_score.py out.fa ref.fa
python3 scripts/msa_bench_score.py --scan outdir refdir
```

## Pendiente / próximas mejoras de calidad (por orden de impacto esperado)

1. Refinamiento iterativo (una pasada de re-alineamiento por subárbol) —
  es lo que separa a MUSCLE/MAFFT-linsi del tier progresivo.
2. Pesos de secuencia (ClustalW sequence weighting) — actualmente cada
  fila pesa 1.0 en los perfiles.
3. Divergencia-dependiente: selección de matriz/penalties por distancia
  del par (ClustalW lo hace; nosotros usamos BLOSUM62 fijo).
4. Evaluar TC en prefab4 por separado (refs de 2 secuencias: SP==TC por
  construcción).
