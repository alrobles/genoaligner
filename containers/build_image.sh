#!/usr/bin/env bash
# Build an apptainer image for genoaligner.
#
# Usage:
#   containers/build_image.sh compile|compute [output.sif]
#
# Why a wrapper and not a bare `apptainer build`: apptainer is present on
# every box we use but its build mode differs (setuid vs user-namespace vs
# fakeroot), and the pull needs a writable cache. This script probes what the
# host supports and picks the most reliable mode, so the same command works
# on reumanlab, reumanlab-alpha and the KU login node.
set -euo pipefail

RECIPE="${1:-}"
OUT="${2:-}"
case "$RECIPE" in
    compile|compute) ;;
    *) echo "usage: $0 compile|compute [out.sif]" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF="${HERE}/genoaligner-${RECIPE}.def"
OUT="${OUT:-${HERE}/genoaligner-${RECIPE}.sif}"

command -v apptainer >/dev/null 2>&1 || {
    echo "apptainer not found" >&2; exit 1; }

# A writable cache is the most common build failure on shared/CI boxes.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${TMPDIR:-/tmp}/apptainer-cache-$(id -u)}"
mkdir -p "$APPTAINER_CACHEDIR"

echo "=== genoaligner image build ==="
echo "recipe : $DEF"
echo "output : $OUT"
echo "apptainer: $(apptainer --version)"
echo "cache  : $APPTAINER_CACHEDIR"
echo "host   : $(hostname)"

# Probe build modes in order of reliability, fall back gracefully.
MODES=()
if [ "$(id -u)" -eq 0 ]; then
    MODES=(--fakeroot "--fakeroot")
elif apptainer build --fakeroot --force /dev/null /dev/null >/dev/null 2>&1; then
    : # unreachable in practice; fakeroot is checked below
fi

build_with() {
    local label="$1"; shift
    echo
    echo "--- attempt: ${label} ---"
    if apptainer build "$@" --force "$OUT" "$DEF"; then
        echo "BUILD_MODE: ${label}"
        return 0
    fi
    return 1
}

# 1) fakeroot (works without root when user namespaces + newuidmap allow it)
if build_with "fakeroot" --fakeroot; then exit 0; fi
# 2) plain (works when the starter is setuid-root)
if build_with "plain" ; then exit 0; fi

echo
echo "All build modes failed. On a host without setuid apptainer and without" >&2
echo "fakeroot privileges, build the .sif on a node that has one (KU HPC does)." >&2
exit 1
