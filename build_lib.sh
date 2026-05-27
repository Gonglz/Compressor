#!/bin/bash
# Build script for momentum compressor C library

set -e

echo "🔨 Building momentum compressor C library..."

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_EXT="dylib"
    EXTRA_FLAGS="-dynamiclib"
else
    LIB_EXT="so"
    EXTRA_FLAGS="-shared"
fi

# Configuration
SRC_DIR="/home/exouser/compressor/final"
BUILD_DIR="$SRC_DIR/build"
OUTPUT_LIB="$SRC_DIR/libmomentum_compressor.$LIB_EXT"

# Find SZ3 library
SZ3_LIB_PATH=""
if [ -f "$HOME/.appfl/.compressor/SZ3/lib/libSZ3c.so" ]; then
    SZ3_LIB_PATH="$HOME/.appfl/.compressor/SZ3/lib"
    SZ3_INCLUDE_PATH="$HOME/.appfl/.compressor/SZ3/include"
    echo "✅ Found SZ3 at: $SZ3_LIB_PATH"
fi

# Compiler flags
CFLAGS="-O3 -Wall -fPIC -std=c99"
LDFLAGS="-lm -lzstd"

# Add SZ3 flags if available
if [ -n "$SZ3_LIB_PATH" ]; then
    CFLAGS="$CFLAGS -DUSE_REAL_SZ3 -I$SZ3_INCLUDE_PATH"
    LDFLAGS="$LDFLAGS -L$SZ3_LIB_PATH -Wl,-rpath,$SZ3_LIB_PATH -lSZ3c"
fi

# Check for blosc
if pkg-config --exists blosc 2>/dev/null; then
    echo "✅ Found blosc"
    CFLAGS="$CFLAGS -DUSE_BLOSC $(pkg-config --cflags blosc)"
    LDFLAGS="$LDFLAGS $(pkg-config --libs blosc)"
elif [ -f "/usr/lib/x86_64-linux-gnu/libblosc.so" ]; then
    echo "✅ Found blosc in system"
    CFLAGS="$CFLAGS -DUSE_BLOSC"
    LDFLAGS="$LDFLAGS -lblosc"
fi

# Create build directory
mkdir -p "$BUILD_DIR"

# Compile
echo "🔧 Compiling with flags: $CFLAGS"
gcc $CFLAGS $EXTRA_FLAGS \
    -o "$OUTPUT_LIB" \
    "$SRC_DIR/momentum_compressor_final.c" \
    $LDFLAGS

if [ $? -eq 0 ]; then
    echo "✅ Build successful: $OUTPUT_LIB"
    
    # Copy to APPFL compressor directory
    APPFL_DIR="/home/exouser/compressor/final/EB-FaLCom/src/appfl/compressor"
    if [ -d "$APPFL_DIR" ]; then
        cp "$OUTPUT_LIB" "$APPFL_DIR/"
        echo "✅ Copied library to: $APPFL_DIR"
    fi
    
    # Show library info
    echo ""
    echo "📊 Library info:"
    ls -lh "$OUTPUT_LIB"
    if command -v ldd &> /dev/null; then
        echo ""
        echo "📚 Dependencies:"
        ldd "$OUTPUT_LIB"
    fi
else
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "✅ Done! Library ready to use with FalComC.py"
