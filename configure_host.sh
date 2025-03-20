#!/bin/bash

# Exit on error
set -e

# Create build directory
mkdir -p build_host
cd build_host

# Use the paths from the original Makefile or environment variables
KOKKOS_ROOT=${KOKKOS_ROOT:-/Users/beckingsale1/code/kokkos/build}
UMPIRE_DIR=${UMPIRE_DIR:-/Users/beckingsale1/opt/umpire}

echo "Using Kokkos at: $KOKKOS_ROOT"
echo "Using Umpire at: $UMPIRE_DIR"

# Configure with CMake
cmake \
  -DENABLE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=${CXX:-/opt/homebrew/Cellar/llvm/19.1.7_1/bin/clang++} \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DKokkos_ROOT="${KOKKOS_ROOT}" \
  -DUMPIRE_DIR="${UMPIRE_DIR}" \
  ..

# Build
make -j

echo ""
echo "Build complete! Run with:"
echo "  ./build_host/umpire_space"