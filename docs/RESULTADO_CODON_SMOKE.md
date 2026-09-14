# RESULTADO — codon_smoke: grid multi-herramienta sobre verdad simulada

**Fecha:** 2026-09-14 · **Rama:** `devin/codon-msa` · **Datos:**
`/beegfs/a474r867/phylogenyAI/data/codon_smoke/` (un workdir por
instancia: `sim_in.fasta`, `sim_true.fasta`, `manifest.json`,
`events.tsv`, output por herramienta, `results.tsv`, `run.log`)

## Instrumento

`scripts/codon_sim_v2.py` — simulador de evolución a nivel codón con
verdad tracked: árbol balanceado, sustituciones codónicas con ω,
indels de codón completo, frameshifts 1-2 nt (`fs_frac`), stops
prematuros (`stop_frac`), fragmentación 5'/3' (`frag5/frag3`), errores
de lectura (`err`, `lr_frac` low-reliability). Cada residuo lleva
`(base, columna_verdadera, codon_id)` → el MSA verdadero se construye
por columna, no por referencia externa.

`scripts/codon_bench_grid.py` — una invocación = una (celda, réplica):
genera la instancia y corre las 4 herramientas sobre el mismo input:

- `genomsa --codon` (CPU shim; bit-idéntico a GPU, probado MI210+A100)
- `macse -prog alignSequences -max_refine_iter 0` (la variante init —
  la comparación honesta contra nuestro single-pass)
- `prank -codon -F` (con el árbol guía verdadero — comparación justa
  estilo Löytynoja & Goldman)
- `threestep` = MAFFT-AA sobre traducción frame-0 → back-translación
  (análogo TranslatorX; filas con stop interno → all-gap, el modo de
  fallo documentado)

`scripts/codon_smoke_grid.json` — 17 celdas × 10 reps: easy /
moderate / divergent / indel-hi / fs-{0.1,0.5,1,3} / stop-{5,20} /
frag / noisy / pseudo-mix / scale-{n512,L5k} / mito-gc2 / small-n32.

`scripts/codon_smoke_report.py` — agrega results.tsv → SPS/TC/wall por
celda×herramienta con IC95 bootstrap (10k resamples).

## Correcciones al scorer (importante — cambian números previos)

Commit `2d86c47`. Dos bugs en `codon_sim_v2.py`:

1. **Absent-absent SPS inflation.** `sps_pairs` contaba un par como
   correctamente co-alineado cuando *ambas* secuencias tenían el
   residuo ausente en el output (`None == None`). Una herramienta que
   emite filas all-gap recibía crédito por residuos que nunca alineó.
   Ahora un par solo cuenta si ambos residuos están presentes y caen
   en la misma columna; los pares omitidos quedan en el denominador.
2. **TC fragilidad.** `tc_score` hacía `break` en el primer residuo
   ausente → `have` sub-contaba según el orden de las filas y `tct`
   podía llegar a 0 → `TC=nan`. Ahora el residuo ausente marca la
   columna como no reproducida sin truncar el conteo.

Más: regex del score-line tolera `nan` (en `codon_bench_grid.py` y en
el rescore), y `scripts/codon_smoke_rescore.py` (nuevo) re-scorea
workdirs desde `manifest.json` + artefactos guardados — sin re-alinear
(`--all`, `--cells`, `--write`).

**Efecto del fix:** filas de genomsa/MACSE/PRANK — SPS idéntico al 4º
decimal (no omiten residuos); TC mueve ~0.001. Filas de threestep —
colapsan de ~0.85-0.95 a ~0.000-0.006: sus números anteriores eran
casi puro artefacto.

## Hallazgo central: el input es mayormente ORF-roto

Fracción de secuencias con stop interno en frame-0 (r0 por celda):

| celda | seqs con stop / n | filas all-gap en ts_got |
|---|---|---|
| easy (stop_frac=0) | 118/128 | 118 |
| frag | 123/128 | 123 |
| fs-0.1 | 128/128 | 128 |
| pseudo-mix | 128/128 | 128 |
| stop-5 | 119/128 | 119 |

El sim genera stops por sustitución codónica aunque `stop_frac=0` —
es el régimen degradado para el que existe MACSE. Consecuencia:
**3-step colapsa en todo el grid**, no solo en las celdas ORF-rotas
explícitas. Caveat para reviewers: un "CDS limpio" real tendría ~0
stops internos; si se quiere ese régimen hay que añadir al sim una
opción que rechace sustituciones creadoras de stop bajo ω<1.
Decisión pendiente: ¿el grid modela input degradado (documentar así)
o hace falta una celda "CDS purificado"?

## Resultados corregidos (SPS vs verdad; IC95 bootstrap; wall s)

13/17 celdas bajo scorer corregido (`rescore_a`, job 29435026:
rescored=470, still-failing=0). fs-0.5, scale-n512, scale-L5k y el
detalle fino de small-n32 quedan para `rescore_b` (ver "en vuelo").

| celda | genomsa | MACSE-init | PRANK -F | 3-step |
|---|---|---|---|---|
| easy | 0.895 (8s) | 0.811 (812s) | **0.975** (3499s) | 0.006 |
| moderate | **0.739** | 0.726 | 0.923 | 0.004 |
| divergent | 0.554 | 0.573 | **0.790** | 0.003 |
| indel-hi | **0.360** | 0.160 | 0.697¹ | 0.000 |
| fs-0.1 | **0.298** | 0.158 | no-output | 0.000 |
| fs-0.5 | **0.153** | 0.126 | no-output | ~0 |
| fs-1 | 0.107 | **0.124** | no-output | 0.000 |
| fs-3 | 0.039 | **0.103** | no-output | 0.000 |
| frag | **0.891** | 0.792 | no-output | 0.001 |
| noisy | **0.887** | 0.797 | 0.975 | 0.004 |
| pseudo-mix | 0.261 | **0.320** | no-output | 0.000 |
| stop-5 | **0.895** | 0.809 | 0.976 | 0.006 |
| stop-20 | **0.893** | 0.806 | 0.976 | 0.003 |
| mito-gc2 | **0.901** | 0.816 | 0.973 | 0.001 |
| small-n32 | 0.946 (2s) | 0.908 (165s) | **0.984** (1282s) | 0.128² |
| scale-n512 | **0.830** (32s) | timeout | — | — |
| scale-L5k | **0.897** (75s) | 1/8 ok (5233s) | — | — |

¹ indel-hi PRANK recuperado parcialmente de artefactos válidos de
corridas previas sobre el mismo input determinista.
² small-n32 threestep: 7/10 con score (3 mafft-fail); el único régimen
donde algunas instancias escapan al colapso.

## Lectura

- **genomsa es la única herramienta con 100% de completitud en todo el
  grid** — todas las instancias, todos los regímenes, 2-75s.
- **PRANK es el líder de precisión donde puede correr** (0.92-0.98 en
  limpio, 0.70-0.79 en difícil) — pero no ingiere input no-%3
  (`no-output` en fs/frag/pseudo-mix) y en el run original tuvo
  timeouts/segfaults en ~30-40% de réplicas limpias. 300-400× más
  lento. Le dimos el árbol verdadero con `-F`.
- **MACSE corre casi todo** pero queda por debajo de genomsa en SPS en
  la mayoría de celdas (solo gana en saturación fs-1/fs-3/pseudo-mix/
  divergent, marginalmente), a 60-100× más walltime. En scale-n512 no
  termina ninguna réplica dentro de 5400s.
- **3-step ≈ 0 en todo el grid** bajo el scorer corregido — el modo de
  fallo "stop interno → fila all-gap" se dispara en >90% de filas en
  todas las celdas. Es el pitch de MACSE cuantificado; y quedaba
  enmascarado por el bug del scorer.

## Claim soportado

*"genomsa es el único alineador de codones robusto a input ORF-roto en
todos los regímenes medidos; supera a MACSE en SPS en la mayoría de
celdas a ~100× menor walltime. Solo PRANK-codon con árbol guía
verdadero lo supera en precisión — en los regímenes donde puede
correr."* NO claim: paridad con PRANK en limpio (pierde ~0.05-0.24
SPS), ni superioridad sobre MACSE en saturación alta.

## En vuelo al cierre de sesión

- `29430948_{134,143,145}` — réplicas scale-n512/r4, scale-L5k/r3,
  scale-L5k/r5 corriendo (MACSE agotó/terminó ~5000-5400s; PRANK en
  curso). Los 12 demás tasks del array completaron (fs-0.5 r3/r8 y
  las 10 de small-n32).
- `29435027` (rescore_b) — `--dependency=afterany:29430948`; corre
  `--all --cells fs-0.5,scale-n512,scale-L5k,small-n32` solo.
- Cadena RF end-to-end: `gsm_iqtr` corriendo (supermatrix genomsa);
  `codon_sm` 29241810 (IQ-TREE codon) PENDING en kbs;
  `rf_cmp`/`macse_pi`/`build_sm`/`sm_iqtre` bloqueados por dependencia.

## Al volver — checklist

1. `squeue -j 29430948,29435027` / leer `rescoreB-*.log` → debe decir
   `still-failing=0`.
2. `python3 scripts/codon_smoke_report.py
   /beegfs/a474r867/phylogenyAI/data/codon_smoke > reporte final`.
3. Verificar fs-0.5/scale-*/small-n32 completas en la tabla.
4. Decidir: ¿celda "CDS purificado" en el sim (rechazo de stops bajo
   ω<1) para responder al reviewer, o documentar el grid como régimen
   degradado?
5. Archivar TWILIGHT_ISSUE_DRAFT.md → issue upstream si se confirma.
