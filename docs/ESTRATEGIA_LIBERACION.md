# genoaligner — Estrategia de liberación y publicación

> **Decisión tomada tras leer los requisitos VIGENTES de JOSS (2026-09-12), no de
> memoria.** Conclusión corta: **JOSS no es elegible hoy y no lo será hasta ~marzo
> 2027 en el mejor caso.** Hay un camino alternativo que sí se puede recorrer ahora.

---

## 1. Por qué JOSS queda descartado por ahora (gates duros y textuales)

JOSS tiene cuatro "pre-review screening criteria"; **fallar una sola es desk
rejection**. genoaligner falla tres:

### Gate 1 — Historial público > 6 meses

> "The repository must have been public for more than six months prior to
> submission... A repository made public immediately before submission, or one
> showing development concentrated into a few days or weeks, will not be accepted.
> **We run automated checks on commit distribution — a repo dump is not a history.**"

> "**Projects developed privately are not eligible until there is a public record of
> open development: at least six months of public history prior to submission**, with
> evidence of releases, public issues and pull requests."

**genoaligner-devel es privado.** Todo el desarrollo (fases 1-8) ocurrió en privado.
Hacerlo público hoy no arregla nada: los checks son automáticos sobre la
distribución de commits, y un repo que aparece con todo el historial de golpe se
lee como "repo dump".

### Gate 2 — Impacto de investigación DEMOSTRADO

> "There must be evidence that the software is being used for research — at minimum
> by the developers themselves, and ideally by others. Acceptable signals include:
> references in published papers or preprints, documented adoption by other research
> groups, or clear integration into research workflows."

> "**Aspirational statements about future use are not sufficient; JOSS will not
> publish papers that are meant to advertise software that is not yet being used in
> research.**"

genoaligner **no tiene usuarios ni uso**. La integración en phylogenyAI está evaluada
y bloqueada (ver `INTEGRACION_PHYLOGENYAI.md`). Este gate es el más duro de los tres.

### Gate 3 — Feature-complete y prácticas abiertas

- Feature-complete: no hay Smith-Waterman, no hay IO FASTA, la API se escribió ayer.
- Prácticas abiertas para proyecto de un autor: "a meaningful public commit history
  over time, tagged releases or a changelog, tests and CI, clear documentation, a
  CONTRIBUTING file". Existen tests y gates, pero **cero historial público**.

### Y una restricción de contenido

> "Your paper must not focus on new research results accomplished with the software."

Un paper JOSS es sobre **el software**, no sobre el hallazgo del `blockDim`. El
hallazgo de Fase 7 (1.5-4.4x, la inestabilidad de 4.4x, el método intercalado) es
material para **otro** venue, no para JOSS.

---

## 2. Qué SÍ es publicable ahora, y dónde

El hallazgo de Fase 7 no necesita un historial público de seis meses: es un resultado
técnico medido. Candidatos honestos:

| venue | qué encaja | requisito que sí cumple |
|---|---|---|
| **arXiv (preprint)** | el hallazgo de ocupación + el método intercalado | ninguno de historial; es el paso natural previo |
| **Workshop de HPC / GPU (SC, PPoPP, o similar)** | "portabilidad HIP con paridad verificada" | revisión por pares, sin gate de historial |
| **JOSS (en ~6 meses, con condiciones)** | el software, cuando haya uso | requiere empezar AHORA el historial público |

---

## 3. Plan de liberación en dos vías

### Vía A — Historial público desde ya (habilita JOSS en ~marzo 2027)

El reloj de los seis meses empieza a contar **cuando el repo se hace público**, no
cuando se envía. Por tanto:

1. **Hacer público `genoaligner` (el de software) AHORA**, con licencia OSI (MIT,
   coherente con el repo de referencia `smarco/WFA`).
2. **Preservar el historial real**, no un dump: el repo ya tiene 60+ commits
   distribuidos en varias sesiones. Publicar el historial completo es legítimo —
   lo que JOSS penaliza es un repo creado de cero con un commit.
   **PERO**: si el historial está comprimido en pocos días, los checks automáticos
   lo notarán. Revisar la distribución real de fechas antes de publicar.
3. Añadir lo que falta para el gate 3: LICENSE, CONTRIBUTING, docs de uso, CI
   (GitHub Actions ejecutando `check_kernel_cpu.sh` — no necesita GPU, encaja
   perfecto en CI).
4. **Conseguir uso real** (gate 2): la vía más rápida es la integración con
   phylogenyAI en su paso de ortología, o publicar el preprint (que cuenta como
   señal de uso si el software se usa para generar sus resultados).
5. `paper.md` + `paper.bib` cuando los gates estén listos.

### Vía B — El hallazgo técnico, ya (arXiv / workshop)

No esperar seis meses para publicar el resultado de Fase 7. El paper técnico sería
sobre: *portabilidad HIP de una sola fuente a dos vendors, con paridad verificada, y
el defecto de ocupación medido (1.5-4.4x, inestabilidad de 4.4x) junto con el método
de medición intercalada.* Ese paper **no** es JOSS (JOSS no publica resultados).

---

## 4. Qué hago con `genoaligner-paper`

El repo `alrobles/genoaligner-paper` es **Overleaf-managed** (base "Initial Overleaf
Import"). Reglas que aplican:

- **Solo fast-forward**; nunca force-push.
- **El usuario compila y pull-ea él; yo no compilo por él.**
- Base canónica: Overleaf.

**Decisión:** usarlo para la **Vía B** (el paper técnico), no para JOSS por ahora.
El `paper.md` de JOSS, cuando toque, iría en el repo de software (JOSS lo exige ahí).

---

## 5. Recomendación

**No escribir el paper JOSS ahora.** Sería producir un artefacto no enviable — el
error que `PREPRINT_ESTADO.md` ya documentó y que esta lectura de requisitos confirma
con los gates concretos.

**Sí hacer ahora, en este orden:**
1. Revisar la distribución de fechas de commits de `genoaligner-devel` (decide si el
   historial sirve para el gate 1).
2. LICENSE + CONTRIBUTING + CI + docs de uso (prepara el gate 3, es trabajo útil
   independientemente de JOSS).
3. Decidir con el usuario: ¿hacer público el repo de software ahora para arrancar el
   reloj de seis meses? Es una decisión con coste (el código se vuelve visible) y el
   usuario es quien la toma.
4. En paralelo, la Vía B se puede escribir cuando quiera.

---

## 6. Evidencia

- Requisitos de JOSS: leídos de https://joss.readthedocs.io/en/latest/submitting.html
  el 2026-09-12 (no de memoria; los criterios cambiaron respecto a lo que asumía
  `PREPRINT_ESTADO.md`, que no mencionaba el gate de seis meses ni el de impacto).
- Formato del paper: Markdown, `paper.md` + `paper.bib`, en el repo del software.
