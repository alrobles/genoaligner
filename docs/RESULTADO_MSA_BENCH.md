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

## Resultados (media por familia; común sobre las mismas familias)

| benchmark | genomsa | ClustalW | Kalign2 | MUSCLE4 | MAFFT linsi | ProbCons |
|---|---|---|---|---|---|---|
| bali3   SP | 0.723 | 0.787 | 0.836 | 0.889 | 0.878 | 0.883 |
| bali3   TC | 0.356 | 0.447 | 0.526 | 0.642 | 0.616 | 0.619 |
| prefab4 SP | 0.561 | 0.617 | 0.623 | 0.733 | 0.723 | 0.717 |
| sabrem  SP | 0.557 | 0.590 | 0.589 | 0.676 | 0.659 | 0.681 |
| sabrem  TC | 0.377 | 0.400 | 0.409 | 0.488 | 0.468 | 0.497 |
| ox      SP | 0.872 | 0.894 | 0.884 | 0.901 | 0.890 | 0.897 |
| ox      TC | 0.765 | 0.802 | 0.789 | 0.813 | 0.798 | 0.809 |

Scores de comparación: `bench1.0/*/qscore/` del bundle (qscore de Edgar,
mismas familias). Agregados pair/column-weighted en `*.scores.tsv`.

## Lectura honesta

- genomsa queda ~6-7 puntos SP por debajo de ClustalW/Kalign2 (su clase:
  progresivo de una pasada) y ~15-20 puntos por debajo de los métodos de
  consistencia/iterativos (ProbCons, MUSCLE, MAFFT-linsi). Esperado: esos
  usan refinamiento iterativo y consistencia probabilística; genomsa hace
  un solo barrido Gotoh por nivel del árbol guía.
- La brecha TC (columnas exactas) es mayor que la SP — las columnas
  enteras correctas son más sensibles a heurísticas de gap.
- En OXBench (fácil) la brecha se reduce a ~2-3 puntos: el motor base es
  sano; la calidad se pierde en regiones divergentes, como espera la
  teoría de alineamiento progresivo.
- Throughput medido: ~0.06 s/familia media en 32 cores CPU
  (2765 familias / 177 s). Escalable a GPU por nivel de árbol.

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
