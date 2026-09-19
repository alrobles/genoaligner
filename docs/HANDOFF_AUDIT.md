# Handoff for an external code audit — genoaligner

**Audit target:** `alrobles/genoaligner-devel` (PRIVATE).
**Commit at handoff:** `173470c`
**Date:** 2026-09-12
**Language:** code, comments, docs and commit messages are in English. This document is in
English so it can be read alongside them; the project owner is a Spanish speaker.

---

## 1. What this software is, in four sentences

`genoaligner` is a biological pairwise sequence aligner that runs on **AMD (ROCm/HIP) and
NVIDIA (CUDA) GPUs from a single source tree**. Every comparable GPU aligner available is
CUDA-only, so the contribution being claimed is **portability with verified parity**, not
throughput — the numbers in the README are explicitly 2-3x below the CUDA state of the
art. It is a research library, not a product: there are no users yet, and that is stated
openly rather than implied.

**The one property the whole project is built around:** a result that looks plausible but
is wrong is treated as the worst possible outcome, worse than an error. Most of the work
below is verification infrastructure aimed at that, and most of the interesting findings
are cases where something *looked* like a result while measuring something else.

---

## 2. Repository layout, and what to trust

```
include/genoaligner/
  api.hpp                      public API (score + CIGAR), the only header a consumer needs
  io/fasta.hpp                 header-only FASTA reader
  backend/wfa_kernel.hip       WFA edit-distance kernel (score + traceback)
  backend/wfa_score_flat.hip   occupancy fix for the score kernel (measured, NOT merged)
  backend/sw_kernel.hip        Smith-Waterman declaration + params
  backend/sw_kernel_impl.hip   Smith-Waterman body (warp-per-row; see §5)
src/api/api.cpp                API implementation: packing, launch, CIGAR decode, validation
bench/tcus.cpp                 throughput probe: per-phase timing, --verify, kernel selection
tests/{parity,api,io,real,concurrency,sw}/   the gates (see §4)
tests/parity/hip_cpu_shim/     host shim that executes the SHIPPED kernel bodies on CPU
tests/data/                    REAL mtDNA (human NC_012920.1, chimp NC_001643.1)
docs/                          MASTERPLAN, ROADMAP, and per-phase result documents
scripts/check_kernel_cpu.sh    the CPU gate — the single most useful entry point
scripts/*.sbatch               Slurm jobs for the KU cluster (ROCm/CUDA/bench/gates)
```

**Trust ranking, for an auditor with limited time:**
1. `scripts/check_kernel_cpu.sh` — runs in ~1 minute, no GPU required. It is the gate that
   was used to catch most defects. **Run it first.**
2. `tests/` — the gates themselves. Each one is meant to be *failable*: it should break if
   the thing it claims to check is broken. Several are documented as having failed on
   purpose at least once.
3. `docs/RESULTADO_*.md` — measured results with the raw numbers and the retractions.
4. `docs/MASTERPLAN.md` — intent. **Least trustworthy of the four**: it records what was
   planned before measurement, so where it disagrees with a RESULTADO document, the
   RESULTADO document is correct.

---

## 3. What is verified, and how

| area | claim | how to check |
|---|---|---|
| Score algebra | 3000 cases identical to edlib | `check_kernel_cpu.sh` (no GPU) |
| Score on GPU | 203/203 re-score, 203/203 well-formed CIGARs | jobs on the cluster, `scripts/h4_*` |
| CIGAR | validated by re-scoring + well-formedness, and by edlib | stages 2-3 of the CPU gate |
| Real sequences | 33 pairs of real mtDNA vs the CPU DP | `tests/real/`, needs a GPU |
| Public API | 200/200 vs CPU DP on GPU; rejects bad input | job 29213867, `tests/api/` |
| Concurrency | 4 threads, 48 real pairs, all correct | job 29226292, `tests/concurrency/` |
| Deployment | clean clone → install → **external consumer runs** | job 29226409 |
| Throughput | ranges with per-GPU spread, single job per GPU | `docs/RESULTADO_B6_TCUPS.md` |
| Smith-Waterman | 8/8 vs independent reference on GPU, blockDim=1 | job 29226705 |

**Six GPUs, two vendors:** MI210 (gfx90a), A100 (sm_80), RTX PRO 6000 (sm_120), plus L40,
V100, RTX 6000 measured in an earlier phase.

---

## 4. Findings worth reading before you review the code

These are the defects that were found by the project's own gates. They are listed because
they show **where the bugs actually live** — which is the most useful prior an auditor can
have here.

1. **A benchmark reported a throughput for work the kernel never did.** The WFA score
    kernel returns early when the true distance exceeds `smax`. A throughput computed as
   `full-DP-cells / time` is meaningless in that case. The probe now reports the resolved
   fraction and refuses to print a number it cannot verify (`bench/tcus.cpp`, exit codes 5/6).
2. **The CIGAR path never worked through the public API, and its test passed anyway.** The
   4th kernel argument is also the dynamic shared-memory size; the API passed the *score*
   kernel's size instead. Every `with_cigar=true` call failed to launch. The test passed
   because its pairs were short enough that the wrong size happened to be accepted.
3. **`smax > 511` was silently clamped**, dropping diagonals without an error, so a caller
   could receive plausible but incorrect alignments. Now rejected as `invalid_argument`.
4. **The library could not be used by a consumer.** Wrong include form for an installed
   tree, backend headers not installed, and `GENOALIGNER_GPU_ARCH` unset by default meant a
   build on a GPU-less login node produced a library that failed at runtime with
   `invalid device function`.
5. **The WFA `default` kernel is temporally non-deterministic.** Seven runs doing
   *identical* work (same distance, same wavefronts walked, same resolved count) varied
   2.5x. Cause: `blockDim = 2*smax+1` leaves most threads idle on barriers.
   `docs/RESULTADO_B6_TCUPS.md` has the raw runs.
6. **Two GPU faults in the SW kernel, invisible to the CPU gate:** host pointers copied
   inside a device struct (fault at an address in the host stack), and static/dynamic
   shared memory overlapping.

**The recurring pattern, stated plainly:** the kernel algebra has been verified
exhaustively; the defects have been in the *layers around it* — the API, the harness, the
build, the documentation. An audit that focuses only on the kernels will find little. The
layers above them are where the risk is.

---

## 5. Known limitations — stated, not hidden

- **`smax <= 511`.** The block mapping needs `blockDim >= 2*smax+1`; beyond that the API
  refuses rather than degrading.
- **Smith-Waterman is incomplete and currently SLOW.** The kernel is correct at
  `blockDim=1` (one thread per block) and **unverified for `blockDim > 1`**: within a row
  the recurrence needs values from the left column, and a per-row barrier does not make
  them visible across threads. A warp-per-row version using `__shfl_up` is **in progress
  and not committed** — the working tree at `173470c` contains the sequential version
  only. `docs/PLAN_SMITH_WATERMAN.md` has the design and the success criteria.
- **SW has no traceback** (score and end coordinate only).
- **No Smith-Waterman in the public API yet** — `align_batch()` is WFA-only.
- **The `flat` kernel is measured but not merged.** It is 1.4-4.4x faster than `default`
  and lacks its non-determinism, but it is score-only and there is no production consumer.
- **A100 timings are unattributed** (10-37% spread, indistinguishable from a busy node).
- **The comparison against Accelign uses their published figure**, not a measurement of
  their binary.

---

## 6. Specific things an auditor should try to break

In rough order of expected yield:

1. **Try to make `align_batch` return a wrong score.** Input validation, empty inputs,
   `pattern_len`/`text_len` inconsistent with the buffer, `smax` at the boundaries (0, 1,
   511, 512), a batch mixing tiny and huge pairs. The `cigar_cap` bug lived exactly here.
2. **Try to make the CIGAR inconsistent with the score.** Re-score every emitted CIGAR;
   check well-formedness; compare against edlib. The WFA traceback bug lived here.
3. **Check whether the gates can fail.** Break something on purpose — invert a comparison
   in a kernel, truncate a buffer size — and confirm the relevant gate goes red. A gate
   that stays green is worse than no gate, because it manufactures confidence.
4. **Audit the shim's honesty.** `tests/parity/hip_cpu_shim/` executes the real kernel
   bodies on host memory. It has **no barrier semantics** (`__syncthreads` is a no-op) and
   no notion of static/dynamic shared-memory overlap. Both SW GPU faults were invisible to
   it. Any conclusion drawn from the shim alone is suspect for anything involving
   concurrency or shared-memory layout.
5. **Check the claim-to-evidence chain in the README.** Every number there should be
   traceable to a job ID or a document. Numbers were published twice in this project and
   retracted twice; if a figure cannot be traced, that is a finding.
6. **Look for resource leaks in `src/api/api.cpp`.** Every early-return path must free what
   was allocated. The error paths were rewritten during this work and deserve a second pair
   of eyes.

---

## 7. How to run things

**CPU gate, no GPU, ~1 minute — start here:**
```bash
git clone https://github.com/alrobles/genoaligner-devel.git && cd genoaligner-devel
bash scripts/check_kernel_cpu.sh
```
It prints `COMPLETE` or `COMPLETE (partial)` and **lists the stages that were skipped**,
so a green run on a GPU-less host cannot be mistaken for full coverage.

**Install (the commands in README.md, verified from a clean clone):**
```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER=$(command -v hipcc) \
      -DGENOALIGNER_BACKEND=rocm -DGENOALIGNER_GPU_ARCH=gfx90a \
      -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build build -j && cmake --install build
```
`-DGENOALIGNER_GPU_ARCH` is **required** when building on a host with no visible GPU.

**GPU work** requires the KU cluster (Slurm, `mi210`/`a100`/`pro6000` partitions). The
`.sbatch` scripts under `scripts/` are self-documenting and each states what it verifies.
Cluster notes: `/beegfs/a474r867/genoaligner/` holds the clone and the logs; the module
setup is `source /kuhpc/sw/lmod/lmod/init/profile` then `ml load compiler/gcc/14.2
cmake/3.30.3 rocm/6.4.3`.

---

## 8. What NOT to change without asking

- **`push`/force-push rules.** `genoaligner` (public, empty) is the future production repo;
  `genoaligner-paper` is Overleaf-managed and must only ever be fast-forwarded, with the
  owner doing the compilation. Do not push to either from this audit.
- **Do not "fix" the retracted numbers or the retraction notes.** The records of what was
  wrong and why are intentional and are considered part of the deliverable.
- **Do not merge the `flat` kernel** into the production path; it is a measured candidate
  awaiting a consumer, and the decision is the owner's.
- **Do not re-derive the WFA recurrence** in `wfa_kernel.hip`. It is a transcription of the
  published algorithm with the source line cited in the file; correctness was established
  against three external oracles and editing it invalidates that.
