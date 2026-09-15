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

## Causa raíz de los stops ubicuos (sim v2.2, commit `cff458b`)

Las sustituciones **nunca** crean stops (`sub_codon` salta candidatos
stop). La fuente era `ins_codons`: insertaba **bases aleatorias
puras** → cada codón insertado tenía 3/64 ≈ 4.7% de ser stop; ~60
codones insertados acumulados por hoja ⇒ ~2-3 stops por secuencia en
TODAS las celdas. Confirmado: easy/r0 v2.1 → 118/128 frame-0 stops;
tras el fix → **0/128**.

**v2.2:** `ins_codons` muestrea del set `sense`. Las celdas quedan:
- limpias de verdad: easy, moderate, divergent, indel-hi, mito-gc2,
  scale-*, small-n32 (3-step ahora compite ahí)
- degradadas por diseño: fs-* (frameshifts), frag (truncamiento rompe
  el frame → len%3≠0), stop-* (stops inyectados), noisy (err puede
  crear stops — realista), pseudo-mix (todo)

**v2.2 run (FINAL):** array `29435703` (tasks 0-169) **COMPLETED
170/170** → `/beegfs/a474r867/phylogenyAI/data/codon_smoke_v22/`
(680 filas = 170 instancias × 4 herramientas). Reporte agregado en
`codon_smoke_v22/REPORT_v22.md` (job 29435925). Scorer corregido
desde el inicio; build atómico pre-compilado. El dataset v2.1 queda
intacto en `codon_smoke/` como referencia del régimen artificial.

## Resultados v2.2 (SPS vs verdad; IC95 bootstrap; wall s; n=10/celda)

| celda | genomsa | MACSE-init | PRANK -F | 3-step |
|---|---|---|---|---|
| easy | 0.889 (8s) | 0.904 (659s) | **0.974** (2929s, 7/10) | 0.937 (108s) |
| moderate | 0.741 (8s) | 0.811 (744s) | **0.931** (2755s, 9/10) | 0.849 (64s) |
| divergent | 0.538 (10s) | 0.620 (825s) | **0.779** (2925s, 8/10) | 0.661 (56s) |
| indel-hi | **0.337** (11s) | 0.329 (1035s) | 0/10 (rc=-9) | **0.497** (106s) |
| fs-0.1 | 0.287 (11s) | 0.311 (1034s) | 0/10 | **0.337** (112s) |
| fs-0.5 | 0.190 (13s) | **0.249** (997s) | 0/10 | 0.079 (110s) |
| fs-1 | 0.117 (15s) | **0.191** (1041s) | 0/10 | 0.016 (119s) |
| fs-3 | 0.051 (19s) | **0.136** (976s) | 0/10 | 0.000 (116s) |
| frag | 0.884 (6s) | **0.898** (455s) | 0/10 | 0.096 (69s) |
| noisy | 0.881 (7s) | 0.900 (587s) | **0.976** (2975s, 9/10) | 0.482 (85s) |
| pseudo-mix | 0.265 (11s) | **0.418** (786s) | 0/10 | 0.002 (92s) |
| stop-5 | 0.889 (7s) | 0.904 (530s) | **0.976** (3124s, 9/10) | 0.847 (85s) |
| stop-20 | 0.888 (7s) | 0.904 (533s) | **0.976** (3099s, 10/10) | 0.590 (86s) |
| mito-gc2 | 0.898 (7s) | 0.918 (558s) | **0.973** (3219s, 10/10) | 0.940 (88s) |
| small-n32 | 0.948 (2s) | 0.956 (79s) | **0.986** (605s) | 0.956 (6s) |
| scale-n512 | **0.818** (33s) | 0/10 (rc=-9) | 0/10 (rc=-9) | 0.806 (10s) |
| scale-L5k | 0.892 (76s) | **0.910** (5160s, 4/10) | 0/10 (rc=-9) | 0.884 (177s) |

Completitud: **genomsa 170/170 · 3-step 170/170 · MACSE 154/170 ·
PRANK 72/170.** PRANK: no-output en todo input no-%3 (fs-*/frag/
pseudo-mix), rc=-9 timeout en indel-hi y scale-*, rc=-11/-9 esporádico
en limpio. MACSE: falla solo en scale (n512 0/10, L5k 6/10 timeout).
genomsa y 3-step: 0 fallos en todo el grid.

## Lectura v2.2 — lo que cambia

- **En celdas limpias genomsa queda 4º de 4 en SPS.** El orden real es
  PRANK > 3-step > MACSE > genomsa (easy/moderate/divergent/mito/
  small-n32). El déficit vs MACSE es chico en easy/mito (~0.015) pero
  grande en divergent (0.082) y moderate (0.070).
- **En celdas degradadas MACSE gana en SPS en todas** (fs-*, frag,
  noisy, pseudo-mix, stop-*): el modelo codónico explícito de MACSE
  maneja frameshifts mejor que genomsa (fs-3: 0.136 vs 0.051; pseudo-
  mix: 0.418 vs 0.265). En v2.1 genomsa "ganaba" esas celdas — era
  artefacto de los stops fantasma que rompían el modelo de MACSE.
- **3-step es un baseline fuerte en input limpio** (supera a genomsa
  en 6/8 celdas limpias, a veces más rápido: scale-n512 10s vs 33s),
  pero **colapsa a SPS≈0 en input con frame roto** (frag 0.096,
  fs≥0.5 →0.00-0.08, pseudo-mix 0.002): traduce en frame-0 y degrada
  a all-gap lo que no traduce. Completa todo, pero su "completitud" es
  hueca en régimen degradado.
- **genomsa es la única herramienta cuyo output es a la vez completo
  (170/170) e informativo en todo el grid** — MACSE falla en escala,
  PRANK no ingiere 80/170 instancias, 3-step emite ruido en degradado.
- **Velocidad: genomsa 2-76s** vs MACSE 455-5160s (~60-80×) vs PRANK
  605-3219s (~300-400×). 3-step es comparable (6-177s).
- **Único nicho donde genomsa gana en SPS:** scale-n512 (0.818 vs
  0.806 de 3-step — los únicos dos que terminan) y empate técnico con
  MACSE en indel-hi.

## Claim soportado (v2.2 — revisado a la baja)

*"genomsa es el único alineador que produce output informativo en el
100% del espectro de calidad de input (limpio → ORF-roto → escala
n≥512), a 60-400× menor walltime que los métodos codon-aware
(MACSE/PRANK). El costo es un déficit de SPS de ~0.01-0.15 vs MACSE
donde MACSE puede correr."*

NO soportado por estos datos: paridad o superioridad de precisión vs
MACSE/PRANK/3-step en régimen limpio — genomsa es el menos preciso del
cuarteto ahí. El claim "MACSE-class accuracy" del preprint hay que
reformularlo como throughput/robustez, o mejorar el alineador
(opciones: iteración de refinamiento tipo MACSE, modelo de penalización
frame-aware más explícito en fs-*).

Nota de interpretación: la diferencia genomsa–MACSE es mayor donde el
modelo evolutivo importa (divergent, fs-*, pseudo-mix) — consistente
con que genomsa usa heurística seeded sin HMM filogenético ni modelo
de codón explícito en la etapa de alineamiento.

## Etapa 2: refine + local-frame tokenización (v2.2-lf, commit `e423c3c`)

Tras el run v2.2 se implementó la arquitectura de dos etapas discutida:
etapa 1 = backbone de codones (existente), etapa 2 = refinamiento
NT-level estilo MACSE. Dos componentes nuevos, ambos host-side:

- **`codon_refine`** (`--refine N`): DP de fase por fila sobre el
  footprint de etapa-1 — la asignación codón→columna queda congelada
  (`pi[k]`), el DP solo elige el límite en nt crudos por slot con
  transiciones `{3 in-frame, 4/5 = ins fs 1-2nt → columna-evento,
  1/2 = fill parcial, 0 = slot vacío}`. SPS-neutro por construcción
  (la membresía de columnas no cambia); emite los eventos fs como
  O(n·K·banda) + merge O(n·ancho): tras paralelizar filas
  (parallel_for) y rehacer el merge (buckets por frontera, no por
  fila×frontera — era O(n²·C)), el refine cuesta 0.4-1.5 s en fs-1 y
  scale-n512 — no es cuello en ningún régimen medido.
- **`codon_encode_local`** (`--local-frame`): DP 1-D que particiona la
  secuencia en bloques de codón y bloques fs de 1-2 nt (fuera del
  stream de tokens). Las filas con fs interno mantienen los tokens
  downstream en-frame → el anclaje de etapa-1 deja de envenenarse.
  Costo de evento `codon_fs_enc=80` (> `stop_pen=60`): un stop
  aislado por error de lectura nunca paga el dodge; un fs real gana
  porque la cola desplazada acumula ≥2 stops. Implica `--refine ≥1`.

**Resultados negativos documentados** (descartados tras medir): el
realineamiento libre por fila y el bandeado con movimiento de columnas
— ambos degradan SPS en TODO régimen (la reubicación greedy por fila
prefiere columnas densas no-homólogas; p.ej. easy 0.893→0.863 incluso
con banda pequeña y perfil limpio). El footprint congelado es el
diseño correcto.

**Run:** array `29437271`, 170/170 tareas, `--tools genomsa
--genomsa-args=--local-frame`, out `codon_smoke_v22_lf/` (mismos seeds
→ inputs idénticos a v2.2). SPS medio por celda (n=10):

| celda | v2.2 base | lf | ΔSPS | ΔTC |
|---|---|---|---|---|
| easy | 0.8895 | 0.8895 | +0.000 | +0.000 |
| moderate | 0.7413 | 0.7413 | +0.000 | −0.002 |
| divergent | 0.5377 | 0.5375 | −0.000 | −0.003 |
| indel-hi | 0.3373 | 0.3373 | +0.000 | +0.000 |
| fs-0.1 | 0.2873 | 0.3020 | **+0.015** | +0.003 |
| fs-0.5 | 0.1901 | 0.2292 | **+0.039** | +0.037 |
| fs-1 | 0.1167 | 0.1610 | **+0.044** | +0.061 |
| fs-3 | 0.0507 | 0.0804 | **+0.030** | +0.075 |
| frag | 0.8841 | 0.8832 | −0.001 | −0.004 |
| noisy | 0.8807 | 0.8805 | −0.000 | −0.001 |
| pseudo-mix | 0.2648 | 0.3745 | **+0.110** | +0.084 |
| stop-5 / stop-20 | 0.8887 / 0.8875 | idem | +0.000 | ~0 |
| mito-gc2 / small-n32 / scale-* | idem | idem | +0.000 | +0.000 |

vs MACSE-init en las celdas donde corre: fs-0.5 0.229 vs 0.249 ·
fs-1 0.161 vs 0.191 · fs-3 0.080 vs 0.136 · pseudo-mix 0.375 vs 0.418.
El déficit de SPS se **redujo ~a la mitad** en fs-* y pseudo-mix;
limpio, escala y noisy quedan bit-idénticos; el overhead de la etapa 2
quedó en ~1 s tras el fix del merge (era +33 s en n512 por un merge
O(n²·C), corregido a O(n·ancho) + filas en paralelo).

**Lectura:** la mayor parte del gap en fs-*/pseudo-mix venía de la
tokenización global-frame (un fs a mitad de secuencia dejaba todos los
tokens downstream fuera-de-frame → anclas basura en etapa-1), no del
realineamiento per se. Lo que queda vs MACSE es estructural de la DP
de tokens (columnas compartidas de indels heredados no se reconstruyen
con realineamiento por fila).

**Decisión GPU:** no se porta — medido. El bottleneck resultó ser el
merge O(n²·C) (bug mío), no la DP: corregido a O(n·ancho) y con las
filas en `parallel_for`, el refine completo cuesta **0.43 s en
scale-n512** (vs 33 s previos) y ~1 s en celdas típicas. Un kernel HIP
ahorraría <0.5 s por instancia a costa de ~300 líneas de packing +
paridad. `codon_encode_local` (~ms) tampoco lo necesita. Ambos corren
host-side antes/después de `msa_align`, así que el path GPU hereda la
mejora de SPS sin cambios; si un día n≫512 lo vuelve cuello, la DP de
fase es embarazosamente paralela por fila y el port es mecánico.

**Default:** `--local-frame` queda opt-in por ahora — gana donde hay
frameshifts reales, es exacto en limpio, y noisy quedó neutro con
`fs_enc=80`, pero fs-3 sigue siendo el régimen más débil (0.080 vs
0.136 MACSE). Falta validar en datos reales (31-locus pipeline) antes
de hacerlo default.

## Estado v2.1 (cerrado)

`29430948` (re-runs + small-n32): 15/15 COMPLETED.
`29435026`/`29435027` (rescore_a/b): COMPLETED — toda la tabla v2.1
bajo scorer corregido. v2.1 queda como dataset del régimen
"todo-degradado" (stops accidentales ubicuos); sus números absolutos
no son comparables a v2.2 por el cambio de distribución de input.

## Pendientes

- Cadena RF end-to-end: `gsm_iqtr`/`codon_sm`/etc. — revisar cola.
- Decidir: ¿mejorar genomsa (refinamiento/frame-model) para cerrar el
  gap vs MACSE, o aceptar el framing throughput/robustez?
- TWILIGHT_ISSUE_DRAFT.md — sin enviar.
