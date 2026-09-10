# Containers (apptainer) — genoaligner

Phase 1 verification, 2026-09-10.

## Why containers, in one paragraph

The first bare-metal iteration of H1 cost **five failed jobs**, none of them
about the science: `cmake` not on `PATH` (the module silently no-ops), a
`GLIBCXX_3.4.32` ABI mismatch, CMake treating `.hip` as a linker input, `hipcc`
being ignored because `CXX_COMPILER` was set as a *target* property, and a job
landing on a CPU node for lack of `--gres`. Every one of those is an
**environment** failure, not a code failure. A container moves the environment
into an artifact we version, copy and diff. That is the entire argument.

## Result: 3/3 hosts PASS, with identical toolchain versions

The compile image builds and runs on every host tried. This is the measured
replicability claim:

| Host | OS | apptainer | build mode | HIP version | cmake | Verdict |
|---|---|---|---|---|---|---|
| reumanlab       | Ubuntu 24.04.4 | 1.5.0 | fakeroot | 6.4.43484-123eb5128 | 3.28.3 | **PASS** |
| reumanlab-alpha | Ubuntu 24.04.4 | 1.5.0 | fakeroot | 6.4.43484-123eb5128 | 3.28.3 | **PASS** |
| KU HPC login    | RHEL 9 (el9)   | 1.3.6 | fakeroot | same image | 3.28.3 | **PASS** |

    KU:      SIF built on the login node, unprivileged, %test PASSED,
             4.9 GB at /beegfs/a474r867/genoaligner/images/genoaligner-compile.sif
    local:   no apptainer on the terminal -> SKIP (an honest gap, not a failure)

**The identical HIP and cmake versions across two different OS families and two
different apptainer majors is the point.** The toolchain stops being a property
of the host and becomes a property of the image.

## The site facts that shape the design (measured, not assumed)

| Fact | Consequence |
|---|---|
| KU HPC **already ships apptainer** (1.3.6) and builds **without root** | No need to build an image elsewhere and ship it in. KU builds its own. |
| KU starter is **not setuid** (user-namespace mode) | Unprivileged builds work. HITP confirmed with the real recipe. |
| reumanlab / alpha have **apptainer 1.5.0** + `fakeroot` + newuidmap | Local builds proven, including a root-owned file written from `%post`. |
| reumanlab has **Docker group access** | Documented fallback build path if a host ever lacks fakeroot. |
| Local NVIDIA GPUs are **not usable** (reumanlab: driver/NVML mismatch; alpha: no `nvidia` module loaded) | Local boxes are **build hosts**, not run hosts. No driver work needed. |
| KU has **no HIP shim for NVIDIA** | The NVIDIA path must come from the image, not from the node. |

## Two images, two roles

    genoaligner-compile.sif   compilers only. NO GPU required.
                              Builds and verifies anywhere -> the replicability metric.

    genoaligner-compute.sif   ROCm userland for execution. Needs --rocm + a driver.
                              This is what makes the MI210 fleet usable.

The split matters: it lets us claim and *measure* replicability using cheap CPU
boxes, while keeping the GPU-dependent artifact honest about its prerequisites.

## Why the compile image does not need a GPU

It carries **compilers** (`hipcc`, `nvcc`), not **runtime libraries**. At build
time nothing touches a device; at run time the host driver is bound in
(`--rocm` binds `/dev/kfd`, `/dev/dri`; `--nv` binds `libcuda`). A compiler does
not need to see the hardware it emits code for — it needs a target and a
`--cuda-path`. This is why `hipcc` lives happily in a container on a machine
whose NVIDIA driver is broken.

## Why this targets the NVIDIA layer specifically

Bare metal failed because KU has no `hipcc` on NVIDIA nodes and no HIP shim in
the module tree. Options checked, with outcomes:

    hip-nvcc on PyPI .................... does not exist
    hipcc on conda-forge ................ exists, but frozen at 6.3.3 (too old)
    hipify-clang / hipify-perl .......... present on KU (fallback, ugly)
    container with a complete hipcc ..... the route under test (R1)

Inside a container we control the entire toolchain, so "is the shim installed on
this node" stops being the question. The image is ours.

## Build

    containers/build_image.sh compile           # -> containers/genoaligner-compile.sif
    containers/build_image.sh compute

The wrapper probes build modes (fakeroot, then setuid) and sets a writable
`APPTAINER_CACHEDIR`, the most common build failure on shared boxes.

## Run

    # compilation, no GPU needed:
    apptainer exec genoaligner-compile.sif hipcc --version

    # execution on MI210 (Slurm node):
    apptainer exec --rocm --bind /beegfs:/beegfs genoaligner-compute.sif \
        ./build/container/genoaligner_probe

## Replicability check

    containers/replicate_check.sh local
    containers/replicate_check.sh reumanlab
    containers/replicate_check.sh reumanlab-alpha

Produces the same PASS/FAIL verdict on every host. GPU availability is reported
but does **not** gate the result — a CPU box passing is a valid data point for a
compile-only image.

## Pitfalls found while doing this

- `apptainer build --fakeroot` needs a **writable cache**; set
  `APPTAINER_CACHEDIR` explicitly or builds fail on shared filesystems.
- The base image is ~4.9 GB. Budget the pull once per host.
- On KU, `fakeroot` prints "User not listed in /etc/subuid, trying root-mapped
  namespace" and then works anyway. That warning is not an error.
- `/etc/singularity/` lingering on reumanlab-alpha produces a scary migration
  notice. Cosmetic.
- A `%test` that only checks `compiler --version` is nearly worthless; the
  compute image's `%test` builds and *runs* a kernel on purpose.
