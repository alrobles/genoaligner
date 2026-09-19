# Fase 8 — Preprint: estado y plan

> **Entregable de Fase 8.** El masterplan pide "Preprint (JOSS o BMC Bioinformatics)".
> Este documento evaluá **qué es publicable hoy** y qué falta. Conclusión: **hay
> material para JOSS en contenido, pero no en requisitos de envío todavía.**

---

## 1. La contribución real (lo que sí es publicable)

**Una implementación de Wavefront Alignment que corre en AMD y NVIDIA desde una sola
fuente HIP, con paridad verificada contra oráculos externos.**

Lo que la sostiene, todo medido en este proyecto:

| afirmación | evidencia |
|---|---|
| una fuente, dos vendors | `wfa_kernel.hip` compila con hipcc y con nvcc (`-D__HIP_PLATFORM_NVIDIA__`) |
| paridad exacta | 6 GPUs (MI210, RTX6000, A100, V100, L40, PRO6000), 100% vs DP de CPU |
| contra oráculos externos | edlib + rapidfuzz + SeqAn3, 0 desacuerdos, ambos backends |
| traceback (CIGAR) | verificado en ambos backends, gate edlib sobre salida de GPU |
| rendimiento | 0.32-1.63 TCUPS en 6 GPUs; optimización 1.5-4.4x |
| replicabilidad | 3 hosts, mismo toolchain; imagen apptainer sin GPU para compilar |

**El hueco que llena**: ninguna herramienta de alineamiento con GPU soporta AMD. La
flota de ~81 MI210 de KU queda ociosa para este trabajo. genoaligner la habilita con
la misma base de código que usa NVIDIA.

**El hallazgo técnico reutilizable**: el defecto de `blockDim = 2*smax+1` (76-97% de
hilos ociosos → inestabilidad de hasta 4.4x y hasta 4.4x más lento). Y el método:
comparación **intercalada** porque la varianza entre nodos es 3-4x.

## 2. Por qué JOSS y no un journal

JOSS publica **software**, no descubrimientos: exige (a) repositorio **público**,
(b) licencia OSI, (c) documentación de uso, (d) tests, (e) que el software "cumpla
una función de investigación". genoaligner encaja en (e) y tiene (d) — pero **hoy no
cumple (a), (b) ni (c)**:

    (a) repo PRIVADO (alrobles/genoaligner-devel)
    (b) sin LICENSE
    (c) sin API pública ni documentación de uso (ver INTEGRACION_PHYLOGENYAI.md §4)

**Un journal como BMC Bioinformatics pediría además una aplicación científica con
resultados biológicos**, que este proyecto todavía no tiene (la integración en
phylogenyAI es Fase 8 sin ejecutar).

## 3. Qué falta para enviar a JOSS (lista verificable)

1. **Repositorio público** (o al menos un espejo público con historia limpia).
2. **LICENSE** OSI (MIT/Apache-2.0) — el masterplan menciona el repo de referencia
   `smarco/WFA` como MIT; elegir una y añadirla.
3. **API pública** (`api.hpp`) + un ejemplo de uso en el README.
4. **Documentación de uso**: hoy el README describe el estado, no cómo usarlo.
5. **Tests ejecutables por un tercero**: `check_kernel_cpu.sh` sirve y no necesita
   GPU — es un activo fuerte para JOSS, porque el revisor puede correrlo.
6. **Zenodo DOI** al hacer el release.
7. `paper.md` + `paper.bib` (JOSS son ~1000 palabras).

## 4. `paper.md` — borrador

Estructura de JOSS, redactada con los datos medidos. **No enviable** hasta cerrar §3.

---

### Summary

genoaligner is a sequence alignment library that builds from a single HIP source
tree for both AMD (ROCm) and NVIDIA (CUDA) GPUs. Existing GPU alignment tools are
CUDA-only; on clusters with AMD accelerators they cannot run at all. genoaligner
targets that gap with one codebase and one set of verification artefacts, so the
same kernels are exercised on both vendors.

### Statement of need

Wavefront Alignment (WFA) computes edit distance in O(ns). GPU implementations
exist but require CUDA. Clusters with AMD Instinct accelerators — the KU cluster
operates ~81 MI210 GPUs — cannot use them for this workload. Portability, not peak
throughput, is the design goal.

### Implementation

The score kernel is a verbatim transcription of the published WFA recurrence
(arxiv/WFA reference implementation), not a re-derivation: the offset macros and
the compute-next step cite their source lines. Portability is achieved through
ROCm's `nvidia_detail` header layer, so `nvcc` compiles the same `.hip` source with
no hipify step.

### Verification

Correctness is established at three levels, in increasing strength:
1. an in-repo O(nm) CPU dynamic program,
2. a CPU shim that executes the shipped kernel body on host memory (runs in seconds,
   needs no GPU), and
3. three independent external oracles — edlib, rapidfuzz and SeqAn3 — on both
   backends, with 0 disagreements.

Six GPUs from two vendors (MI210, RTX 6000, A100, V100, L40, RTX PRO 6000) were
measured with identical verification, and the traceback (CIGAR) path is verified
against edlib on GPU-emitted output.

### Performance

Kernel throughput is 0.32-1.63 TCUPS across the six GPUs, 2-3x below a
state-of-the-art CUDA aligner measured on the same card. A block-occupancy fix
(decoupling block size from the search bound) gave 1.5-4.4x and removed a
non-determinism of up to 4.4x between launches.

### Acknowledgements / References

References to be added in `paper.bib`: WFA (Marco-Sola et al.), edlib, SeqAn3,
Accelign, MMseqs2-GPU.

---

## 5. Decisión

Fecha: 2026-09-12. **No se envía nada todavía.** El contenido está; los requisitos de
envío (repo público, licencia, API, docs de uso) no. Prioridad: **API pública** —
desbloquea JOSS, la integración (§INTEGRACION) y cualquier uso real.

Este documento evita el peor resultado posible: escribir un preprint para un
software que un revisor no puede instalar ni ejecutar.
