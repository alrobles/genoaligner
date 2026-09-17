#!/bin/bash
set -euo pipefail

ROOT=${PHYLOGENY_ROOT:-/beegfs/a474r867/phylogenyAI}
ASTER_DIR=${ASTER_DIR:-$ROOT/tools/ASTER-v1.25}
ASTER_REPO=${ASTER_REPO:-https://github.com/chaoszhang/ASTER.git}
ASTER_COMMIT=${ASTER_COMMIT:-db2b3e95da5bb0318b933afe1a144eb943ef7cbf}
CXX=${CXX:-g++}
PROFILE=${CASTER_BUILD_PROFILE:-portable}

case "$PROFILE" in
    portable)
        FLAGS=(-std=gnu++17 -O3 -ffast-math -pthread)
        ;;
    strict)
        FLAGS=(-std=gnu++17 -O2 -fno-fast-math -ffp-contract=off -pthread)
        ;;
    native)
        FLAGS=(-std=gnu++11 -march=native -Ofast -pthread)
        ;;
    *)
        echo "unknown CASTER_BUILD_PROFILE: $PROFILE" >&2
        exit 2
        ;;
esac

mkdir -p "$(dirname "$ASTER_DIR")"
if [ ! -d "$ASTER_DIR/.git" ]; then
    git clone "$ASTER_REPO" "$ASTER_DIR"
fi

DIRTY=$(
    git -C "$ASTER_DIR" status --porcelain --untracked-files=all |
        sed '/^?? bin\//d'
)
if [ -n "$DIRTY" ]; then
    echo "ASTER checkout is dirty: $ASTER_DIR" >&2
    exit 1
fi

git -C "$ASTER_DIR" fetch --depth 1 origin "$ASTER_COMMIT"
git -C "$ASTER_DIR" checkout --detach "$ASTER_COMMIT"

test "$(git -C "$ASTER_DIR" rev-parse HEAD)" = "$ASTER_COMMIT"
mkdir -p "$ASTER_DIR/bin"
BIN="$ASTER_DIR/bin/caster-site-$PROFILE"
BIN_TMP="$BIN.tmp.$$"
META="$BIN.build.tsv"
META_TMP="$META.tmp.$$"
trap 'rm -f "$BIN_TMP" "$META_TMP"' EXIT

"$CXX" "${FLAGS[@]}" "$ASTER_DIR/src/caster-site.cpp" -o "$BIN_TMP"
test -x "$BIN_TMP"
mv "$BIN_TMP" "$BIN"

{
    printf 'key\tvalue\n'
    printf 'runtime_backend\tcpu-%s\n' "$PROFILE"
    printf 'profile\t%s\n' "$PROFILE"
    printf 'aster_commit\t%s\n' "$ASTER_COMMIT"
    printf 'compiler\t%s\n' "$("$CXX" --version | sed -n '1p')"
    printf 'flags\t%s\n' "${FLAGS[*]}"
    printf 'sha256\t%s\n' "$(sha256sum "$BIN" | awk '{print $1}')"
} > "$META_TMP"
mv "$META_TMP" "$META"

trap - EXIT
"$BIN" -h 2>&1 | sed -n '1,3p'
echo "CASTER binary: $BIN"
