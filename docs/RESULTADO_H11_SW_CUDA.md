# RESULTADO H11-SW-CUDA — el path SW en NVIDIA, Fase C (deuda cerrada)

Fecha: 2026-09-12 · Rama: main · Commit: c369b42
Job GPU: **29230631** (V100-SXM2-16GB, r31r30n01, nodo idle, `gres=gpu:v100:1`)

## Qué se verificó

El mismo árbol de fuentes que corre en MI210, compilado con **nvcc** (CUDA
12.4) a través de la capa `hip/nvidia_detail` de ROCm — sin hipify, sin shim —
sobre una **Tesla V100 (sm_70, warpSize=32 real)**.

La novedad no es "otra GPU": es el **primer run con warpSize=32 real**. Todo
run anterior del score kernel fue warpSize=64 (MI210) o emulación (shim a 32 y
64). El kernel distribuye columnas por lanes y cruza tiles de `warpSize` — el
caso `n=33 (crosses a 32-lane tile)` es exactamente la dependencia que solo un
warp de 32 real prueba.

## Evidencia

| verificación | resultado |
|---|---|
| `sw_score_kernel` (test_sw_gpu) | **210/210** vs referencia, `warpSize=32`, `grid=53 block=128` |
| `sw_trace_kernel` (test_sw_trace_gpu) | **209/209**: score==score-kernel==referencia, CIGAR well-formed, re-score exacto, igual al walk por valores |
| API pública (test_api) | **PASS** — batería SW completa en device: ejemplo documentado, respuestas conocidas, 120/120 vs CPU ref, `too_large`, todos los rechazos |
| **SeqAn3** | **201/201** filas sobre el emit del trace kernel + **150/150** sobre el emit de la API — 0 discrepancias de score en ambos |

Notas menores: el oráculo saltó 8 filas con campos vacíos (casos de score 0 o
secuencia vacía — formato TSV, no discrepancia). `cuobjdump` confirma cubins
`sm_70` embebidos.

## Estado de la portabilidad después de este run

- SW score + trace + API pública: **verificado en AMD MI210 (warp 64) y NVIDIA
  V100 (warp 32)** — ambos desde el mismo árbol de fuentes.
- La afirmación "portable" para SW ya tiene evidencia en ambos vendors.
- Sigue sin evidencia: rendimiento SW en cualquier plataforma (Fase D).
