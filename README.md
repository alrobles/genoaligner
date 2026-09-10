# genoaligner

Hardware-agnostic (CUDA + ROCm) sequence alignment library.

**Status:** development (private). Phase 0 feasibility verified; design decision pending.

## Why this exists

Every GPU sequence-alignment tool available in 2026 is CUDA-only, including the
current state of the art (MMseqs2-GPU, ~102 TCUPS on 8x L40S, Nature Methods
2025; Accelign, 9-16 TCUPS on RTX PRO 6000, BMC Bioinformatics 2026). The KU
HPC cluster has ~81 AMD Instinct MI210 GPUs (gfx90a) that no alignment tool can
use. genoaligner targets portability: one HIP codebase that compiles and runs
on both CUDA and ROCm, so the AMD fleet stops being idle for this workload.

The goal is NOT to beat Accelign on raw TCUPS. The goal is to produce results
identical to a CPU reference (SeqAn/Parasail) while running on hardware no
existing tool supports.

## Verified environment (KU HPC, 2026-09-10)

    ROCm 6.4.3   hipcc 6.4.43484, AMD clang 19   -> smoke test PASSED on MI210
    CUDA 13.0    nvcc at /kuhpc/sw/nvhpc/.../cuda/13.0
    hipify-clang and hipify-perl present at /kuhpc/sw/rocm/6.4.3/bin/
    GCC 11.5 / 14.2, CMake 3.30.3

Note: Slurm scripts need `#!/bin/bash -l` (or source
/kuhpc/sw/lmod/9.3/init/bash) because the non-interactive shell does not
initialise Lmod.

## Docs

- [docs/OPTION_A_ANALYSIS.md](docs/OPTION_A_ANALYSIS.md) — build-from-scratch
  analysis: real code-size measurements from WFA-GPU/Accelign, effort
  estimate, pros/cons, and the architecture recommendation (pure HIP).
