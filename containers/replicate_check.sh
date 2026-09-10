#!/usr/bin/env bash
# Replicability probe: build and run the genoaligner container on a host.
#
# Usage: containers/replicate_check.sh [host]   (default: local)
#
# The point of this script is NOT to build fast. It is to produce the same
# PASS/FAIL verdict on every machine, so "replicable" is a measured claim with
# N data points instead of an adjective.
#
# It reports, for the host: apptainer version, build mode that worked, whether
# the compile image builds, and whether the built binary runs. GPU execution is
# reported separately and is allowed to be UNAVAILABLE without failing the
# script -- that is a real and useful result (the compile image does not need a
# GPU by design).
set -uo pipefail

HOST="${1:-local}"

if [ "$HOST" = "local" ]; then
    RUN=(bash)
else
    RUN=(ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" bash -s)
fi

REMOTE_SCRIPT='
set -uo pipefail
echo "host        : $(hostname)"
echo "os          : $(. /etc/os-release; echo $PRETTY_NAME)"
echo "kernel      : $(uname -r)"
echo "apptainer   : $(apptainer --version 2>/dev/null || echo NONE)"
[ -x "$(command -v apptainer)" ] || { echo "RESULT: SKIP (no apptainer)"; exit 0; }

# GPU visibility: informational only, must not gate the result.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "gpu(nvidia) : $(nvidia-smi -L | head -1)"
elif [ -e /dev/kfd ]; then
    echo "gpu(amd)    : /dev/kfd present"
else
    echo "gpu         : UNAVAILABLE (fine for a compile-only image)"
fi

# Which uid does an apptainer container see? This is the fakeroot/user-ns probe.
uid_in=$(apptainer exec docker://alpine:3.20 id -u 2>/dev/null || echo ERR)
echo "uid in ctr  : $uid_in"

echo "--- build compile image ---"
work=$(mktemp -d)
trap "rm -rf $work" EXIT
cat > "$work/ga.def" <<EOF
Bootstrap: docker
From: rocm/dev-ubuntu-24.04:6.4.3-complete
%post
    set -e
    (command -v dnf >/dev/null && dnf install -y cmake gcc-c++ make which || apt-get update && apt-get install -y cmake g++ make) >/dev/null 2>&1
    mkdir -p /opt/genoaligner
%runscript
    hipcc --version | head -1
    cmake --version | head -1
EOF
export APPTAINER_CACHEDIR="$work/cache"; mkdir -p "$APPTAINER_CACHEDIR"
if apptainer build --fakeroot --force "$work/ga.sif" "$work/ga.def" >/dev/null 2>&1; then
    echo "build mode  : fakeroot"
elif apptainer build --force "$work/ga.sif" "$work/ga.def" >/dev/null 2>&1; then
    echo "build mode  : plain"
else
    echo "build mode  : FAILED"
    echo "RESULT: FAIL (could not build)"
    exit 1
fi

echo "--- run image ---"
if apptainer run "$work/ga.sif" 2>&1 | sed "s/^/  /"; then
    echo "RESULT: PASS (compile image builds and runs on this host)"
else
    echo "RESULT: FAIL (image built but did not run)"
    exit 1
fi
'

printf '%s\n' "$REMOTE_SCRIPT" | "${RUN[@]}" 2>&1
