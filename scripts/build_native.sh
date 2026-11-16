#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT_DIR/cpp/sa_core.cpp"
BINARY="$ROOT_DIR/artifacts/bin/sa_core"
# Build stamp: sha256 of the exact source bytes compiled, flags and compiler.
# The Python bridge (src/adaptive_3dbp/native/cpp_bridge.py) writes the same
# format and rebuilds when it no longer matches.
STAMP="$BINARY.sha256"
LOCK="$BINARY.lock"
# -ffp-contract=off: no FMA contraction, so the decoder reproduces the
# Python reference bit for bit. Keep in sync with COMPILE_FLAGS in cpp_bridge.py.
FLAGS="-O3 -std=c++17 -ffp-contract=off"
mkdir -p "$ROOT_DIR/artifacts/bin"

# Same lock file the bridge flock()s (flock(1) is missing on macOS: skip there).
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK"
  flock 9
fi

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"; rm -f "$BINARY.tmp.$$" "$STAMP.tmp.$$"' EXIT
# Compile a private copy of the bytes that are hashed.
cp "$SOURCE" "$BUILD_DIR/sa_core.cpp"
if command -v sha256sum >/dev/null 2>&1; then
  DIGEST="$(sha256sum "$BUILD_DIR/sa_core.cpp" | cut -d' ' -f1)"
elif command -v shasum >/dev/null 2>&1; then
  DIGEST="$(shasum -a 256 "$BUILD_DIR/sa_core.cpp" | cut -d' ' -f1)"
else
  DIGEST="$(python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$BUILD_DIR/sa_core.cpp")"
fi
COMPILER_ID="$(g++ --version | head -n 1 | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

# shellcheck disable=SC2086 # FLAGS is a list of separate arguments
g++ $FLAGS "$BUILD_DIR/sa_core.cpp" -o "$BINARY.tmp.$$"
printf 'source_sha256 %s\nflags %s\ncompiler %s\n' "$DIGEST" "$FLAGS" "$COMPILER_ID" > "$STAMP.tmp.$$"
mv -f "$BINARY.tmp.$$" "$BINARY"
mv -f "$STAMP.tmp.$$" "$STAMP"
echo "Native core built at $BINARY (source sha256 $DIGEST)"
