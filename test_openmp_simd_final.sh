#!/usr/bin/env bash
# Compile and run the optimized OpenMP/SIMD final variant against captured data.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-$ROOT_DIR/dataset/resnet50}"
ROUNDS="${ROUNDS:-3}"
THREADS="${OMP_NUM_THREADS:-8}"

SZ3_PREFIX="${SZ3_PREFIX:-$HOME/.appfl/.compressor/SZ3}"
SZ3_INCLUDE="$SZ3_PREFIX/include"
SZ3_LIB="$SZ3_PREFIX/lib"

export LD_LIBRARY_PATH="$SZ3_LIB:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

TEST_BIN="$ROOT_DIR/test_openmp_simd_final"

echo "Compiling test binary: $TEST_BIN"
gcc -std=c99 -O3 -march=native -ffast-math -fopenmp \
    -I"$ROOT_DIR" -I"$SZ3_INCLUDE" \
    "$ROOT_DIR/test_v21_full.c" \
    "$ROOT_DIR/momentum_compressor_openmp_simd_final.c" \
    -L"$SZ3_LIB" -Wl,-rpath,"$SZ3_LIB" \
    -lSZ3c -lzstd -lm \
    -o "$TEST_BIN"

echo "Running optimized variant"
echo "Dataset: $DATASET"
echo "Threads: $OMP_NUM_THREADS"
echo "Rounds expected by test_v21_full: $ROUNDS"

"$TEST_BIN" v22sz3 "$DATASET"
