# Análisis de la Opción A — genoaligner desde el inicio

**Repo:** `alrobles/genoaligner-devel` (privado)
**Fecha:** 2026-09-10
**Pregunta:** ¿por qué dije "6 meses" y es eso correcto?

---

## 0. Corrección de mi propia estimación

Dije "meses de desarrollo" de forma perezosa, sin datos. Fui a medirlo.
**La estimación estaba mal, y en la dirección que tú intuías: es menos.**

Un alineador GPU no es un proyecto de compilador. Es un **kernel** más un
arnés. Los números reales:

| Componente | WFA-GPU (referencia) | Accelign (referencia) |
|---|---|---|
| Kernel de alineación | **688 líneas** (`sequence_alignment_kernel.cu`) | ~2,000 líneas útiles |
| Lógica de alto nivel | 962 (`align.cu`) + 470 (`sequence_alignment.cu`) | ~1,500 |
| Utilidades (I/O, FASTA) | ~800 (`aligner.c`, `sequence_reader.c`) | reutilizable |
| Total código útil | **~2,900 líneas** | ~4,000 líneas |

El resto de esos repos es **ruido inflado**:
- WFA-GPU: 3,930 de sus 4,275 KB son **datos de test** (`sequences_1000.h`).
- Accelign: 1,957 KB son tablas de tuning **autogeneradas** por arquitectura,
  y ~1,200 KB son variantes duplicadas (`old/`, `_startendpos`, local/semi/global).

**El corazón del problema son ~700-2,000 líneas.** Eso no son 6 meses. Eso son
semanas — con IA en el bucle, menos.

---

## 1. Qué NO partimos de cero (el punto clave de tu intuición)

Partimos con:

1. **El algoritmo está publicado y es público.** WFA (Wavefront Alignment,
   Marco-Sola et al. 2021) y WFA-GPU (Aguado et al. 2023) tienen papers con
   pseudocódigo y formulación matemática completa. No hay que *inventar* el
   método, hay que *implementarlo*.

2. **Una implementación de referencia que leer.** WFA-GPU es MIT: puedo leer
   el kernel exacto, entender el mapeo a hilos, y reimplementar el diseño en
   HIP con conocimiento de causa. No es plagio — es ingeniería informada por
   una referencia abierta.

3. **`hipify-clang` y `hipify-perl` en el cluster.** Verificado:
   `/kuhpc/sw/rocm/6.4.3/bin/hipify-clang`. Convierte CUDA→HIP
   automáticamente. Para el *scaffolding* (alloc de memoria, streams, launch),
   hace el 80% del trabajo.

4. **El arnés ya existe en phylogenyAI.** Tengo el pipeline, el dataset
   (19,336 secuencias), los benchmark de MAFFT con timings. La validación
   está montada.

5. **El toolchain funciona.** Fase 0 demostró que HIP compila y corre en la
   MI210 hoy.

**No es "desde cero". Es "desde el algoritmo publicado + una referencia
abierta + el toolchain funcionando".**

---

## 2. Estimación realista por fases

| Fase | Trabajo | Líneas estimadas | Días (con IA) |
|---|---|---|---|
| 1 | Scaffolding: build system dual (CUDA/ROCm), CI, estructura | ~400 | 2-3 |
| 2 | Kernel Smith-Waterman affine (score-only) en HIP | ~500 | 3-5 |
| 3 | Traceback (la parte difícil — memoria y sincronización) | ~600 | 5-8 |
| 4 | Capa de portabilidad: un código, dos backends | ~300 | 2-4 |
| 5 | Validación vs SeqAn/Parasail (paridad exacta) | ~200 | 2-3 |
| 6 | Optimización (sequence packing, tiling) | ~500 | 5-10 |
| 7 | Integración en pipeline + benchmark | ~300 | 2-4 |
| **Total** | | **~2,800** | **~25-35 días** |

**No 6 meses. 5-7 semanas** de trabajo efectivo, con margen para lo que se
descubra por el camino (traceback y sincronización son donde aparecen
sorpresas).

Y hay entregables intermedios publicables:
- Fase 2-3 completas → kernel funcional + paridad validada (paper técnico)
- Fase 4 → el framework portable (el hueco científico real)

---

## 3. Pros de la Opción A (implementar desde el inicio)

**Pros:**

1. **Diseño limpio desde el principio para portabilidad.** Si portas código
   CUDA existente con hipify, heredas la arquitectura CUDA como deuda técnica.
   Escribir desde el inicio con abstracción HIP te da un kernel que es
   *nativo* en ambas plataformas, no un port.

2. **Control total del camino crítico.** El traceback es donde WFA-GPU y
   Accelign difieren en trucos de memoria. Diseñarlo tú = entiendes cada
   decisión, y puedes optimizar para *tu* caso de uso (filogenética:
   secuencias cortas-medianas, muchas). No para el caso genérico.

3. **Publicable como contribución original.** "Framework de alineación
   portable CUDA/ROCm" es un paper. Un port con hipify es un truco de
   ingeniería, difícil de publicar como novedad.

4. **El código es tuyo, sin ataduras de licencia.** WFA-GPU es MIT (bien),
   pero GASAL2 tiene licencia más restrictiva. Control total = libertad
   comercial y de relicenciamiento.

5. **Justificación de hardware.** Un alineador propio que corre en las 81
   MI210 justifica las GPUs y la infraestructura ante quien financie.

6. **La IA cambia el cálculo.** Con IA como acelerador de implementación, el
   coste marginal de escribir vs portar se reduce. El kernel de 688 líneas
   no se escribe en 688 iteraciones manuales.

**Contras (honestos):**

1. **El traceback es genuinamente difícil.** Gestión de memoria en GPU,
   sincronización entre celdas, y el mapeo de wavefronts a hilos. Es donde
   WFA-GPU dedica 688 líneas de kernel + 1,400 de soporte. No es trivial.

2. **La optimización es un pozo sin fondo.** Escribir un kernel que FUNCIONE
   es semanas; escribir uno que compita con Accelign (9 TCUPS) es meses. Hay
   que decidir explícitamente dónde parar: **¿necesitas rendimiento máximo o
   necesitas que corra en MI210?** Para tu ventaja competitiva, lo segundo.

3. **Riesgo de "no inventado aquí".** El kernel propio puede acabar 2-5×
   más lento que Accelign en NVIDIA. Eso está bien si el criterio es
   portabilidad, pero hay que saberlo de antemano.

4. **Mantenimiento.** Un alineador propio es software que hay que mantener.
   Un port de algo mantenido por otros es menos carga a largo plazo.

5. **Riesgo de que hipify no cubra el kernel.** Los kernels complejos con
   `__shfl`, memoria compartida dinámica o inline PTX suelen necesitar
   reescritura manual. La capa de scaffolding se convierte, el kernel no.

---

## 4. La decisión real no es A vs B, es el criterio de éxito

Esto es lo importante del análisis. Antes de elegir A, hay que responder:

**¿Cuál es el objetivo del software?**

- **Si es RENDIMIENTO MÁXIMO** → A es mala idea. Nunca ganarás a Accelign
  (equipo con años de optimización) partiendo de cero. Usa B o C.

- **Si es PORTABILIDAD / DESBLOQUEAR HARDWARE** → A es la opción correcta.
  El criterio no es "más rápido que Accelign", es "corre en MI210 donde
  nadie más corre". Y para eso, un kernel propio limpio es mejor que un port.

- **Si es PUBLICAR** → A da un paper de método. B da un reporte técnico.

Mi lectura de tu estrategia ("competencia, capacidad, diseño como el debe ser")
apunta a **portabilidad + soberanía**, no a récord de TCUPS. En ese caso A es
defendible, con una condición: **fijar el techo de optimización por adelantado**
(parar en Fase 6, no perseguir Fase 7 indefinidamente).

---

## 5. Enmienda al plan: arquitectura para A

Si vas con A, la decisión técnica central es **cómo se escribe "una vez,
corre en dos"**:

    Opción 1: HIP puro         → hipcc compila para CUDA y ROCm
             (recomendada)        una base de código, dos backends nativos
    Opción 2: SYCL/oneAPI      → agnóstico por diseño, pero el cluster
                                  solo tiene compiler/intel/25 (faltan libs GPU)
    Opción 3: Kokkos           → portabilidad probada, overhead de abstracción
    Opción 4: hipify del CUDA  → convierte, pero hereda deuda técnica CUDA

**Recomiendo HIP puro.** Es lo que AMD usa (minimap2 heterogéneo, jul 2026),
convierte a CUDA con `hipcc -D__HIP_PLATFORM_NVIDIA__`, y el cluster lo tiene
en versiones 6.1 a 6.4. SYCL sería más elegante pero el cluster no lo tiene
listo. Kokkos añade una capa que no necesitas para dos plataformas.

---

## 6. Preguntas abiertas para la decisión

1. **¿El criterio es "corre en MI210" o "compita con Accelign en TCUPS"?**
   (esto define todo lo demás)

2. **¿WFA (wavefront) o Smith-Waterman clásico (antidiagonal)?**
   - WFA: mejor para secuencias similares (lecturas largas), usado por
     WFA-GPU. Complejidad O(ns) donde s = distancia de edición.
   - SW antidiagonal: más general, mejor paralelismo masivo, usado por
     CUDASW++/GASAL2.
   - Para filogenética (genes homólogos, 50-95% identidad) **WFA es más
     apropiado**: la distancia de edición es baja, así que el coste es bajo.

3. **¿Alcance del primer entregable?** Propongo: kernel score-only que corra
   en MI210 con paridad exacta contra SeqAn. Eso valida todo el diseño sin
   meterse aún en traceback.

4. **¿Deadline o acoplamiento con phylogenyAI?**

---

## 7. Recomendación

**Sí a A, con alcance acotado.** Mi propuesta concreta:

- **Semanas 1-2:** scaffolding dual + kernel score-only en HIP
- **Semanas 3-4:** validación de paridad en MI210 + CUDA
- **Semanas 5-6:** traceback
- **Semanas 7-8:** capa de portabilidad + primer benchmark honesto

Con un criterio de éxito explícito: **"genoaligner corre en MI210 y produce
resultados idénticos a SeqAn"** — no "genoaligner es más rápido que Accelign".

Si eso se cumple, tienes infraestructura propia y una contribución publicable.
Si a mitad de camino se ve que el traceback se atasca, el kernel score-only
ya es un entregable útil y el retroceso a B sigue disponible.

**El riesgo está mitigado por fases con entregables. No hay apuesta única.**
