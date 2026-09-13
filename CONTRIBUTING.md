# Contributing to genoaligner

## Before you change the kernel

The kernels in `include/genoaligner/backend/` are a **verbatim port** of the
published WFA formulation, with the source file and line cited at each site. Do not
"improve" the recurrence, re-derive it, or reorder the terms: every such change in
this project's history produced a wrong answer that looked plausible. If you think
the recurrence is wrong, cite the reference implementation and open an issue first.

## The rule that this project learned the hard way

**Nothing touches a GPU until its CPU gate is green.**

```bash
scripts/check_kernel_cpu.sh      # seconds, needs no GPU, no ROCm
```

This runs the *shipped* kernel body on host memory against an independent O(nm)
dynamic program, then re-runs it under the sanitizer. It has caught bugs that a
cluster job would have reported as a memory fault an hour later.

## Every change needs four things, not one

A new check or kernel variant is only evidence if:

1. it **fails** on the bug it is meant to catch (try it against the bug),
2. it **passes** on correct input,
3. it **declines to pass on no input** (an empty result is not a pass), and
4. it reports the **denominator**: a pass rate without its coverage is not a result.
   `44/44 correct` meant nothing while 170 of 214 pairs produced nothing at all.

## Reporting a measurement

- State which **backend and device** each number came from.
- Report the **spread**, not just the mean. The default score kernel was measured
  oscillating **2.179-9.619 ms within a single process** (4.4x). A mean hides that.
- **Do not compare across jobs.** Node-to-node variance on KU was measured at 3-4x
  on identical input. Use an **interleaved A/B** design (A,B,A,B in one job) and
  report the per-round **ratio**, which cancels the node's state.
- Say what a number does **not** prove. "Parity 100%" over abandoned pairs, or over
  one backend, is a narrower claim than it reads.

## Build and test

```bash
# CPU gate (always, first)
bash scripts/check_kernel_cpu.sh

# the AMD and NVIDIA builds are in docs/DEPLOYMENT_MANIFEST.md
# the public API test:
g++ -O2 -std=c++17 -DGENOALIGNER_HIP_SHIM -Itests/parity/hip_cpu_shim -I. \
    -o /tmp/test_api tests/api/test_api.cpp src/api/api.cpp
```

Build from a **clean clone** before claiming a deployment works: doing that found
two headers that only compiled by accident of include order.

## Style

- Comments explain **why**, especially when the code looks redundant. Several
  barriers and sentinel checks in this codebase look redundant and are not.
- Commit messages state the measurement and the command, not the intent.
- Documentation is bilingual in practice: the code and commits are English, the
  analysis documents (`docs/*.md`) are Spanish. Keep it that way.

## Governance and support

Single-maintainer project. Issues and pull requests are the support channel; there
is no SLA. Security or correctness concerns about the alignment results should be
filed as issues with a minimal reproducer (the `min reproducer 1..3` cases in
`tests/parity/r1_check.cpp` show the expected shape).
