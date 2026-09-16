#!/bin/bash
set -euo pipefail

ROOT=${PHYLOGENY_ROOT:-/beegfs/a474r867/phylogenyAI}
ASTER_DIR=${ASTER_DIR:-$ROOT/tools/ASTER-v1.25}
ASTER_REPO=${ASTER_REPO:-https://github.com/chaoszhang/ASTER.git}
ASTER_COMMIT=${ASTER_COMMIT:-db2b3e95da5bb0318b933afe1a144eb943ef7cbf}
BUILD_JOBS=${BUILD_JOBS:-4}

mkdir -p "$(dirname "$ASTER_DIR")"
if [ ! -d "$ASTER_DIR/.git" ]; then
    git clone "$ASTER_REPO" "$ASTER_DIR"
fi

if [ -n "$(git -C "$ASTER_DIR" status --porcelain)" ]; then
    echo "ASTER checkout is dirty: $ASTER_DIR" >&2
    exit 1
fi

git -C "$ASTER_DIR" fetch --depth 1 origin "$ASTER_COMMIT"
git -C "$ASTER_DIR" checkout --detach "$ASTER_COMMIT"
make -C "$ASTER_DIR" caster-site -j "$BUILD_JOBS"

test "$(git -C "$ASTER_DIR" rev-parse HEAD)" = "$ASTER_COMMIT"
test -x "$ASTER_DIR/bin/caster-site"
"$ASTER_DIR/bin/caster-site" -h 2>&1 | head -3
