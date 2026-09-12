# Fase 8 — Integración en phylogenyAI: evaluación

> **Entregable de Fase 8.** Este documento **evalúa** la integración con evidencia;
> no la promete. Conclusión corta: **hoy no es enchufable, y el bloqueante es la
> falta de API pública, no el rendimiento.**

---

## 1. Qué pide el masterplan

> Fase 8 — Integración en phylogenyAI + paper
> - Enchufar genoaligner al pipeline (donde aporte).

El "donde aporte" es la parte que hay que responder con datos, no con entusiasmo.

## 2. Qué hace phylogenyAI y qué hace genoaligner

**phylogenyAI** (revisado en `~/GitHub/phylogenyAI/PLAN.md`, 2026-09-12):

- Alineamiento: **MAFFT** (`--auto` / `--globalpair`) → alineamiento **múltiple**.
- Ortología: **DIAMOND/MMseqs2** + reciprocal best hit (para ampliaciones futuras).
- Búsqueda similar a BLAST contra NCBI nt: **descartada** (el CSV de MamPhy ya trae
  las accesiones curadas; repetir BLAST redescubre lo ya curado).

**genoaligner** (estado real):

- Alineamiento **pairwise** por distancia de edición (WFA), score + CIGAR.
- Verificado al 100% en 6 GPUs de 2 vendors contra el DP de CPU y 3 oráculos externos.
- **Sin API pública**: los kernels se alcanzan desde los harnesses.

## 3. Dónde encaja, y dónde NO

| paso del pipeline | ¿encaja genoaligner? | por qué |
|---|---|---|
| MSA (MAFFT/MACSE) | **NO** | genoaligner es pairwise. Un MSA no se hace con pares independientes. Prometerlo sería falso. |
| BLASTN contra nt | **NO** | descartado en phylogenyAI §51 por redescubrir lo curado. |
| **Ortología (DIAMOND/MMseqs2)** | **SÍ, conceptualmente** | búsqueda/reclutamiento de similitud = alineamiento pairwise masivo. Es el único encaje real. |
| Limpieza (stop codons, frameshifts) | no evaluado | fuera del alcance de lo medido. |

El único punto de encaje es **ortología por reciprocal best hit**, que es exactamente
un problema de alineamiento pairwise a escala. Y ahí el valor medido es: **corre en
MI210**, donde ni DIAMOND ni MMseqs2-GPU corren (son CUDA-only), y la flota AMD de
KU es ~81 tarjetas.

## 4. El bloqueante real: no hay API

Esto es lo que impide la integración hoy, y no es el rendimiento:

    include/genoaligner/api.hpp     <- previsto en el masterplan §2.2, NO existe
    src/io/fasta.cpp                <- previsto, NO existe
    src/wfa/wfa_host.cpp            <- previsto, NO existe

Sin `api.hpp` no hay forma de que phylogenyAI llame a genoaligner sin copiar código
de los harnesses. El kernel está verificado; la **interfaz de consumo** no existe.

## 5. Qué haría falta, en orden

1. **API pública** (`include/genoaligner/api.hpp`): una función tipo
   `score_pair(pattern, text, smax) -> {score, cigar}` backend-agnóstica, con el
   dispatch a ROCm/CUDA oculto. Es el trabajo de mayor valor y menor riesgo.
2. **IO FASTA** mínimo (`src/io/fasta.cpp`).
3. **Bench de extremo a extremo** con un subconjunto de genes de phylogenyAI: leer
   FASTA, alinear pares, escribir resultados. Solo entonces se puede decir "aporta".
4. **Medir el paso de ortólogos** contra MMseqs2/DIAMOND sobre los mismos datos, en
   el hardware donde ambos corran (NVIDIA) — y en MI210, donde solo corre genoaligner.

## 6. Qué NO afirmar

- **No** decir "genoaligner acelera phylogenyAI". No se ha medido ningún paso del
  pipeline real.
- **No** decir "reemplaza a MAFFT". Es pairwise vs múltiple.
- **No** decir "ya está integrado". No hay API.

## 7. Registro de la decisión

Fecha: 2026-09-12. Se evaluó la integración y se concluyó que **el siguiente paso es
la API pública**, no una conexión al pipeline. Escribir la conexión antes que la API
sería copiar el harness a phylogenyAI — deuda técnica disfrazada de integración.

Este documento existe para que la decisión sea revisable: si alguien quiere integrar
antes, el §4 dice exactamente qué falta.
