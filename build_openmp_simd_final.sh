#!/usr/bin/env bash
# Build the stable OpenMP + compiler-SIMD optimized MomentumCompressor variant.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT_DIR/momentum_compressor_openmp_simd_final.c"
OUT="$ROOT_DIR/libmomentum_compressor_openmp_simd_final.so"
APPFL_COMPRESSOR_DIR="$ROOT_DIR/EB-FaLCom/src/appfl/compressor"

SZ3_PREFIX="${SZ3_PREFIX:-$HOME/.appfl/.compressor/SZ3}"
SZ3_INCLUDE="$SZ3_PREFIX/include"
SZ3_LIB="$SZ3_PREFIX/lib"

if [[ ! -f "$SRC" ]]; then
    echo "Source not found: $SRC" >&2
    exit 1
fi

if [[ ! -f "$SZ3_LIB/libSZ3c.so" ]]; then
    echo "SZ3 library not found at $SZ3_LIB/libSZ3c.so" >&2
    echo "Set SZ3_PREFIX to the directory containing include/ and lib/." >&2
    exit 1
fi

CFLAGS=(
    -std=c99
    -O3
    -march=native
    -ffast-math
    -fopenmp
    -fPIC
    -Wall
    -Wextra
    -I"$ROOT_DIR"
    -I"$SZ3_INCLUDE"
)

LDFLAGS=(
    -shared
    -L"$SZ3_LIB"
    -Wl,-rpath,"$SZ3_LIB"
    -lSZ3c
    -lzstd
    -lm
)

echo "Building $OUT"
gcc "${CFLAGS[@]}" "$SRC" "${LDFLAGS[@]}" -o "$OUT"

echo "Built: $OUT"
ls -lh "$OUT"

if [[ "${1:-}" == "--install" ]]; then
    install -m 0755 "$OUT" "$APPFL_COMPRESSOR_DIR/libmomentum_compressor.so"
    echo "Installed optimized library for FalComC:"
    echo "$APPFL_COMPRESSOR_DIR/libmomentum_compressor.so"
fi
