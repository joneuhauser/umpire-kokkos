#!/bin/bash

# Exit on error
set -e

# Create build directory
mkdir -p build_cuda
cd build_cuda

# Use the paths from the original Makefile or environment variables
KOKKOS_ROOT=${KOKKOS_ROOT:-/Users/beckingsale1/code/kokkos/build}
UMPIRE_DIR=${UMPIRE_DIR:-/Users/beckingsale1/opt/umpire}
CUDA_ARCHS=${CUDA_ARCHS:-"70;75;80"}

echo "Using Kokkos at: $KOKKOS_ROOT"
echo "Using Umpire at: $UMPIRE_DIR"
echo "Using CUDA architectures: $CUDA_ARCHS"

# Check for required CUDA compiler
NVCC_PATH=$(which nvcc 2>/dev/null || echo "")
if [ -z "$NVCC_PATH" ]; then
  echo "Error: nvcc (CUDA compiler) not found in PATH. Please install CUDA Toolkit or add it to your PATH."
  exit 1
fi

# Configure with CMake
cmake \
  -DENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=${CXX:-/opt/homebrew/Cellar/llvm/19.1.7_1/bin/clang++} \
  -DCMAKE_CUDA_COMPILER=${NVCC_PATH} \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DKokkos_ROOT="${KOKKOS_ROOT}" \
  -DUMPIRE_DIR="${UMPIRE_DIR}" \
  ..

# Build
make -j

echo ""
echo "Build complete! Run with:"
echo "  ./build_cuda/umpire_space"