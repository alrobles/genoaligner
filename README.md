# genoaligner

**Pairwise sequence alignment on AMD *and* NVIDIA GPUs, from one source tree.**

Every GPU sequence-alignment tool available in 2026 is CUDA-only, including the
current state of the art. genoaligner compiles the same HIP source with `hipcc`
(ROCm/AMD) and `nvcc` (CUDA/NVIDIA), with no hipify step and no per-vendor fork, so
hardware that no existing tool can use stops being idle for this workload.

The goal is **portability with verified correctness**, not peak throughput.
Correctness is established against an independent CPU dynamic program and three
external oracles (edlib, rapidfuzz, SeqAn3) on both vendor paths.

- **Status:** development. WFA score and traceback (CIGAR) verified on CPU, MI210
  (hipcc) and five NVIDIA GPUs; the public API is tested on device; the production
  checklist is in [docs/PLAN_PRODUCCION.md](docs/PLAN_PRODUCCION.md).
- **Scope:** this is a **pairwise** aligner (edit distance / WFA). It is not a
  multiple aligner and does not replace MAFFT or MACSE, and it is not a search tool
  that recruits candidates from a database.

## Install

Requirements: a HIP compiler (`hipcc` for AMD, `nvcc` + ROCm headers for NVIDIA),
CMake >= 3.18, and a C++17 host compiler.

```bash
cmake -S . -B build \
      -DCMAKE_CXX_COMPILER=$(command -v hipcc) \
      -DGENOALIGNER_BACKEND=rocm \
      -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build build -j
cmake --install build
```

That produces `lib/libgenoaligner.a` and headers under `include/`.

For the NVIDIA path, set the backend and point CMake at CUDA. **CUDA 12.x is
required** (the ROCm compatibility layer does not parse under 13.0), and the version
you need depends on the GPU you target:

| GPU | arch | CUDA |
|---|---|---|
| V100 / RTX 6000 / A100 / L40 | sm_70 / sm_75 / sm_80 / sm_89 | 12.4 works |
| RTX PRO 6000 (Blackwell) | sm_120 | **12.8+** (12.4 cannot target it) |

```bash
cmake -S . -B build -DGENOALIGNER_BACKEND=cuda \
      -DCMAKE_CXX_COMPILER=/path/to/cuda/bin/nvcc \
      -DGENOALIGNER_CUDA_ROOT=/path/to/cuda
```

Full deployment notes, including the failure table and how to verify from a clean
clone: [docs/DEPLOYMENT_MANIFEST.md](docs/DEPLOYMENT_MANIFEST.md).

## Use

```cpp
#include <genoaligner/api.hpp>
#include <genoaligner/io/fasta.hpp>
```

Read a FASTA, align the pairs you want, read the results. The complete runnable
example is [examples/align_fasta.cpp](examples/align_fasta.cpp):

```cpp
std::vector<genoaligner::io::FastaRecord> records;
std::string err;
if (!genoaligner::io::read_fasta_file("seqs.fa", &records, &err)) { /* handle */ }

genoaligner::AlignRequest r;
r.pattern     = records[0].sequence.data();
r.pattern_len = (int)records[0].sequence.size();
r.text        = records[1].sequence.data();
r.text_len    = (int)records[1].sequence.size();
r.smax        = 64;     // max edit distance to search
r.with_cigar  = true;   // false = score only (cheaper, no traceback)

genoaligner::BatchResult batch = genoaligner::align_batch({r});
if (!batch.ok()) { /* device or argument failure -- see batch.error */ }

for (const auto& res : batch.results) {
    if (!res.resolved) continue;      // true distance exceeded smax: NOT an error
    std::printf("score=%d cigar=%s\n", res.score, res.cigar.c_str());
}
```

Link against the library:

```bash
hipcc -O2 -std=c++17 -I$PREFIX/include my_tool.cpp -L$PREFIX/lib -lgenoaligner
```

### Real output

`examples/align_fasta.cpp` on human mtDNA (NC_012920.1), MI210. CIGARs shortened
with `...` for readability:

```
genoaligner rocm on AMD Instinct MI210
read 1 record(s) from tests/data/mtdna_human.fa

aligned 3 pair(s): 3 resolved, 0 unresolved

pair 0: score=1 cigar=MMMM...MMMMXMMMM...MMMM (validated: rescore=yes wellformed=yes)
pair 1: score=1 cigar=MMMM...MMMMXMMMM...MMMM (validated: rescore=yes wellformed=yes)
pair 2: score=1 cigar=MMMM...MMMMXMMMM...MMMM (validated: rescore=yes wellformed=yes)

example complete
```

Each pair differs by exactly one substitution, so `score=1` and a single `X` is
correct — and `rescore`/`wellformed` are the library validating its own CIGAR before
returning it.

## Limits — read before you size a run

- **`smax` must be in `[0, 511]`.** Above that the kernel's thread-to-diagonal mapping
  cannot cover the wavefront and would silently skip diagonals, so the API **rejects**
  out-of-range values instead of degrading. Requests that fail return
  `Status::invalid_argument`.
- **Unresolved is not an error.** A pair whose true distance exceeds `smax` comes back
  with `resolved == false`, `score == -1` and an empty CIGAR. Check
  `resolved_count` on the batch: if most of your input is unresolved, raise `smax`.
- **One `smax` per batch, and it is the MINIMUM** of the requests'. The kernel takes a
  single bound, and a per-request `smax` is a limit *you* set, so the batch never
  grants more than the narrowest request allows. Consequence: one narrow request
  lowers the bound for every pair in the batch. To mix bounds, group by `smax` and
  make several calls.
- **Thread safety.** Concurrent `align_batch()` calls on **disjoint inputs** return
  correct results (verified, 4 threads x 48 real pairs). Two caveats: the API stores
  **pointers**, not copies, so an input buffer must stay alive and unmodified until
  the call returns; and launches go to the default stream, so do not read timing
  numbers from concurrent calls. Details in `include/genoaligner/api.hpp`.
- **Input is not validated as ACGT.** Real FASTA carries `N`, ambiguity codes and
  lowercase soft-masking; the reader preserves bytes as they are.

## Verify

The gate needs no GPU and runs in seconds — it executes the shipped kernel body on
host memory against an independent dynamic program:

```bash
bash scripts/check_kernel_cpu.sh
```

On a GPU node the same script additionally runs the real-sequence, concurrency and
batch-semantics tests; on a CPU-only host those stages report as skipped and say so
in the summary. Building CMake tests produces the same set:

```bash
cmake -S . -B build -DGENOALIGNER_BUILD_TESTS=ON ... && cmake --build build && ctest --test-dir build
```

## Measured performance

Kernel throughput on six GPUs, same source, each verified against the CPU reference.
Kernel-only = cells of resolved pairs / kernel time; end-to-end includes packing and
transfers but excludes one-time context bring-up.

| GPU | arch | kernel-only (TCUPS) |
|---|---|---|
| RTX PRO 6000 | sm_120 | 1.21 - 1.63 |
| L40 | sm_89 | 1.01 - 1.48 |
| A100 | sm_80 | 0.45 - 0.73 |
| V100 | sm_70 | 0.53 - 0.68 |
| MI210 | gfx90a | 0.39 - 0.47 |
| RTX 6000 | sm_75 | 0.32 - 0.43 |

These are honest ranges across three regimes, not best cases. On the same card
(RTX PRO 6000) a CUDA-only state-of-the-art aligner reports 9-16 TCUPS, so this is
**2-3x below it after optimisation** — portability is the contribution, not
throughput. Numbers, regimes and the measurement method (interleaved A/B, because
node variance here is 3-4x): [docs/BENCHMARK_FASE6.md](docs/BENCHMARK_FASE6.md) and
[docs/RESULTADO_FASE7_OPTIMIZACION.md](docs/RESULTADO_FASE7_OPTIMIZACION.md).

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) — the rules this codebase is held to.
- [docs/DEPLOYMENT_MANIFEST.md](docs/DEPLOYMENT_MANIFEST.md) — deploying elsewhere.
- [docs/PLAN_PRODUCCION.md](docs/PLAN_PRODUCCION.md) — production readiness checklist.
- [docs/BENCHMARK_FASE6.md](docs/BENCHMARK_FASE6.md) — six GPUs, two vendors.
- [docs/RESULTADO_FASE7_OPTIMIZACION.md](docs/RESULTADO_FASE7_OPTIMIZACION.md) —
  the block-occupancy fix (1.5-4.4x) and how it was measured.
- [docs/OPTION_A_ANALYSIS.md](docs/OPTION_A_ANALYSIS.md) — why one HIP source.
- [docs/CONTAINERS.md](docs/CONTAINERS.md) — apptainer images and 3-host
  replicability.

## Licence

MIT, with a partial-derivation notice: the wavefront kernels transcribe the
published WFA reference implementation (`github.com/smarco/WFA`, MIT, (c) 2017
Santiago Marco-Sola), cited line by line in the source. See [LICENSE](LICENSE).

## AI usage

Generative AI was used in developing this software, under human direction and
review: code generation and refactoring, test scaffolding, and documentation
drafting. The problem framing, the architectural decisions (one HIP source with a
two-compiler build, verification-first gates, interleaved measurement) and the
validation of all AI-assisted output were done by the human author, who is
responsible for the accuracy of the work. Disclosed here because any venue this
reaches will require it.
