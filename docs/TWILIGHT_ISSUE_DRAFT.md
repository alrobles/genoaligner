# DRAFT — upstream issue for TurakhiaLab/TWILIGHT (not yet filed)

Title: CUDA realignment path is non-deterministic and memory-dependent:
       same input, different trees/MSAs across runs and GPUs

## Summary

On identical inputs and guide tree, TWILIGHT's CUDA path produces
different alignments across runs, and differs from both `--cpu-only`
and the HIP GPU path. The HIP GPU path is byte-identical to `--cpu-only`.
The divergence appears only when the deferred-realignment path
(`fallback2cpu`) activates at scale, which depends on
`availableMem`-derived `numBlocks` — i.e. the result depends on how
much free GPU memory happened to exist at launch.

## Reproduction

Dataset: COI locus, 1608 sequences (mammalian mitochondrial), same
guide tree file, same binary, same node.

| route | alignment width |
|---|---|
| `twilight --cpu-only` (V100 or MI210 host) | 2991 |
| HIP GPU (MI210, gfx90a) | 2991 — byte-identical to `--cpu-only` |
| CUDA GPU (V100-SXM2-32GB) | 5494, and 5382 on a re-run — **non-deterministic** |

- 1608/1608 output rows differ between CUDA-GPU and CPU reference.
- On MI210 only 3/893 subtree profiles were deferred to
  `fallback2cpu`; on V100 **786/893 (88%)** were deferred, because
  `numBlocks` is sized from `availableMem` (70% of free device memory).
- The deferred path (`subtreeAln`/`mergeInsertions`) produces a
  different alignment than the `--cpu-only` pipeline, so the final
  result depends on GPU free memory, not just the input.
- The DP is integer (int16/int32): non-determinism suggests a race in
  the deferred TBB realignment, not floating-point noise.
- On a small input (64 taxa, ~1.5 kb) CUDA==HIP==CPU byte-exact: the
  divergence only emerges when deferral activates at scale.

## Environment

- TWILIGHT 0.2.3, commit f11db22
- V100-SXM2-32GB, nvcc sm_70
- MI210 (gfx90a), ROCm 6.4.3
- KU HPC cluster

## Why it matters

For a phylogenetics tool the alignment must be a deterministic
function of the input; a GPU-memory-dependent result makes
publications hard to reproduce and silently changes results when a
colleague re-runs on a different card or a busier node.

## Suggested directions

1. Make the deferred realignment deterministic and consistent with
   `--cpu-only` (or document the divergence explicitly).
2. Decouple `numBlocks`/batching from `availableMem`, or at least make
   batch boundaries deterministic so results don't depend on free VRAM.
3. Related smaller issues we verified:
   - HIP-only silent fallback: `src/hip/alignment-gpu.hip.cpp` prints
     "CPU on No. X" and recomputes on CPU when the kernel path doesn't
     consume expected columns — masks kernel defects without a failure
     signal. (0 occurrences in our runs, but the mechanism is live.)
   - Build break: `-march=native` in CMAKE_CXX_FLAGS + GCC 11 AMX
     headers (`amxtileintrin.h` uses `__builtin_ia32_*` nvcc doesn't
     define) breaks nvcc builds; `-march=native` is also fragile for
     binaries that run on heterogeneous nodes.
