# Plan de lanzamiento a producción — genoaligner

> **Estrategia decidida por el usuario (2026-09-12):** madurar en
> `genoaligner-devel` (privado) y **promover a `genoaligner` (público) cuando las
> banderas de producción estén en verde**. No se publica software a medias.
>
> **Estado de partida:** la auditoría del 2026-09-12 encontró 5 defectos en la API
> recién escrita. 4 arreglados y verificados en GPU; el 5º (CMake) arreglado y
> **verificado desde entonces**.

---

## 0. Los dos repositorios

    alrobles/genoaligner-devel   PRIVADO   desarrollo, historial completo (86+ commits)
    alrobles/genoaligner         PÚBLICO   VACÍO — destino de la promoción
    alrobles/genoaligner-paper   PRIVADO   draft del paper (Overleaf-managed)

**Regla de promoción:** `genoaligner` público NO recibe trabajo incremental. Se
llena **una vez**, cuando las banderas estén verdes, y desde ahí es el repo de
usuarios. El desarrollo sigue en `-devel`.

---

## 1. Lo que YA está verificado (no requiere trabajo)

| verificación | evidencia |
|---|---|
| API corre en GPU | job 29213867, H8 PASS, MI210, 200/200 vs DP de CPU |
| validación de entrada | smax 512/-1 y punteros nulos rechazados; 511 aceptado |
| `libgenoaligner.a` + install | CMake RC=0, BUILD RC=0, INSTALL RC=0 |
| gate CPU cubre la API | Stage 5 en `check_kernel_cpu.sh` |
| CI sin GPU | `.github/workflows/cpu-gate.yml` (gate + API + clon limpio) |
| 6 GPUs, 2 vendors | `docs/BENCHMARK_FASE6.md` |
| licencia y atribución | MIT + PARTIAL DERIVATION NOTICE (WFA de Marco-Sola) |

**Descubrimiento operativo del sitio** (costó varias sesiones, queda registrado):

    source /kuhpc/sw/lmod/lmod/init/profile      # NO init/bash: MODULEPATH queda vacío
    ml load compiler/gcc/14.2                     # cmake 3.30.3 exige GLIBCXX_3.4.32
    ml load cmake/3.30.3                          # el nombre lleva versión
    # cmake 3.30.3 vive en /kuhpc/sw/cmake/3.30.3/gcc/14.2/bin/cmake

---

## 2. Banderas de producción (gates de promoción)

Cada bandera es un comando que se puede correr y un criterio que se puede fallar.
**Todas deben estar en verde antes de tocar el repo público.**

### B1 — IO FASTA ✅ COMPLETA
- [x] `include/genoaligner/io/fasta.hpp` (header-only, sin dependencia de link).
- [x] Test con multilínea, CRLF, líneas en blanco, descripción, y 4 entradas
      malformadas que deben fallar.
- [x] Criterio cumplido: el caso multilínea da **24 bases donde un lector ingenuo da
      10** — el bug que un test de conteo habría dejado pasar.

**De paso, tres defectos encontrados y arreglados** (todos del mismo tipo: confundir
"no puede correr aquí" con "está roto"):
1. `test_api.cpp` decidía `gpu` en COMPILACIÓN; un binario hipcc en un nodo sin
   device fallaba 10 checks de corrección y anunciaba librería rota.
2. `probe.hip` abortaba con error HIP por lo mismo. Ahora sale 77 ("skipped").
3. Los checks que dependen de ejecución se gatean todos con `check_exec()` y un flag
   `g_dont_care`; los de rechazo de entrada siguen corriendo en cualquier host.

Resultado: en un host sin GPU, `ctest` reporta **Skipped** (no Failed) y `fasta_runs`
pasa. En MI210 (job 29213875) la API da **PASS**. El CI no tendrá un build rojo falso.


### B2 — Test con secuencias biológicas reales ✅ COMPLETA
- [x] Dataset real versionado: **mitocondrias humana (NC_012920.1, 16,569 bp) y de
      chimpancé (NC_001643.1, 16,554 bp)**, de NCBI E-utilities, con procedencia en
      el archivo del test.
- [x] 5 casos: fragmentos reales, **región D-loop (rica en repeticiones)**, divergencia
      real humano-chimp, moléculas completas de 16.5 kb, y la `N` real.
- [x] Criterio cumplido: **33 pares reales, 0 discrepancias contra el DP de CPU,
      0 CIGARs inválidos** (job 29213888, MI210).

**Dos defectos encontrados via el propio test, y ambos arreglados:**
1. El comentario del test afirmaba distancia ~72 para ventanas de 600 bp
   humano-chimp. **Medido: ~300** (genomas que difieren en 15 bp acumulan indels;
   comparar ventanas a offset fijo es ~50% divergente, no ~12%). `smax=200` dando 0
   resueltos era correcto — el comentario mentía.
2. **Agujero real:** toda la sección divergente devolvía 0 resueltos, así que habría
   pasado igual si la resolución estuviera rota. Añadidas ventanas de 100 bp que
   deben resolver (6/6), con aserción explícita.
3. El resumen del gate decía "real sequences verified" **cuando el stage se saltaba**
   por falta de GPU. Ahora dice "COMPLETE (partial)" y enumera lo NO cubierto.


### B3 — Concurrencia / reentrada
- [ ] Decidir y **documentar** si `align_batch` es thread-safe.
- [ ] Si no lo es: decirlo en `api.hpp` (una línea basta, pero debe estar).
- [ ] Si lo es: test con 2+ hilos llamando en paralelo.

**Por qué:** estado global (`shim::smem_vec`, y el device en sí) hace que la
respuesta no sea obvia. Un integrador necesita saberlo.

### B4 — Semántica de batch documentada y probada
- [ ] Hoy el batch toma el **mínimo `smax`** de todas las peticiones y lo aplica a
      todas. Está comentado en el código pero **no en la cabecera pública**.
- [ ] Decidir: ¿mínimo, o agrupar por smax? Documentar en `api.hpp`.
- [ ] Test que verifique el comportamiento elegido.

**Por qué:** un usuario que pase peticiones con smax distintos debe poder predecir
qué obtiene. Ahora mismo obtiene el mínimo, silenciosamente.

### B5 — README de usuario
- [ ] Sección de instalación (CMake, ya verificado).
- [ ] Un ejemplo completo compilable: leer FASTA → alinear → escribir resultados.
- [ ] La tabla de límites (`smax ≤ 511`; abandonados; un smax por batch).
- [ ] El README actual describe el **estado de desarrollo**, no el **uso**.

### B6 — Números de TCUPS con barras de error
- [ ] Re-medir con el diseño **intercalado** (el validado en Fase 7).
- [ ] Reportar media **y** dispersión, por GPU.
- [ ] Criterio: ningún número sin su dispersión, ninguna comparación cruzando jobs.

**Por qué:** el README público citará estos números. Ya nos equivocamos una vez
publicando un 1.93x medido cruzando jobs.

### B7 — Decisión sobre Smith-Waterman
- [ ] Escribir SW, **o** declarar explícitamente en el README que solo hay WFA.
- [ ] Criterio: el README no promete lo que no existe.

### B8 — Saneamiento del historial público
- [ ] **Decisión del usuario:** el historial de `-devel` tiene 86 commits en 3 días.
- [ ] Opciones: (a) promover el historial completo, (b) promover un historial
      limpio y revisado, (c) empezar de cero con un commit fundacional.
- [ ] Criterio: ningún secreto, ninguna ruta privada, ninguna credencial en el
      historial (revisar con `git log -p | grep` antes de publicar).

---

## 3. Procedimiento de promoción (cuando B1-B8 estén verdes)

```bash
# 1. Verificar el árbol que se va a publicar, desde un clon limpio
git clone <devel> /tmp/promote && cd /tmp/promote
bash scripts/check_kernel_cpu.sh          # debe llegar a CPU GATE COMPLETE

# 2. Revisar que no haya nada privado en el historial
git log -p --all | grep -inE "beegfs/|/home/|api[_-]?key|token|password|secret" | head

# 3. Promover (ejemplo con historial completo; ajustar según B8)
git remote add public git@github.com:alrobles/genoaligner.git
git push public main

# 4. En el repo público: LICENSE visible, README de usuario, y tag
git tag -a v0.1.0 -m "genoaligner v0.1.0 — WFA score+CIGAR, ROCm + CUDA"
git push public v0.1.0

# 5. Verificar ESI el repo público: clonar desde cero y correr el gate
git clone https://github.com/alrobles/genoaligner /tmp/es_check
cd /tmp/es_check && bash scripts/check_kernel_cpu.sh
```

**Paso 5 no es opcional.** El repo público es un artefacto distinto del privado: hay
que clonarlo desde fuera y correr el gate, porque ya nos encontramos dos headers que
solo compilaban por accidente de orden de includes.

---

## 4. Orden de trabajo recomendado

| # | bandera | esfuerzo | desbloquea |
|---|---|---|---|
| 1 | B1 IO FASTA | 1 sesión | B2, B5 |
| 2 | B2 secuencias reales | 1 sesión | confianza en producción |
| 3 | B4 semántica de batch | 30 min | B5 |
| 4 | B3 concurrencia | 30 min | documentación honesta |
| 5 | B5 README de usuario | 1 sesión | publicación |
| 6 | B6 TCUPS con barras de error | 1 job | credibilidad |
| 7 | B7 decisión SW | decisión | alcance |
| 8 | B8 saneamiento | decisión | **el último antes de publicar** |

**Un solo bloque de trabajo por bandera, con su verificación, y commit.** Cada una
deja el repo mejor independientemente de si se publica o no.

---

## 5. Lo que NO hará este plan

- No publicar antes de que las 8 banderas estén verdes.
- No prometer Smith-Waterman si no existe.
- No reportar TCUPS sin dispersión.
- No hacer público el historial sin la revisión de B8.
