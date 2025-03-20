#!/bin/bash

# Create build directory
mkdir -p build_hip
cd build_hip

# Use the paths from the original Makefile or environment variables
KOKKOS_ROOT=${KOKKOS_ROOT:-/g/g16/becks/code/kokkos/install}
UMPIRE_DIR=${UMPIRE_DIR:-/g/g16/becks/code/umpire/install}

echo "Using Kokkos at: $KOKKOS_ROOT"
echo "Using Umpire at: $UMPIRE_DIR"

cmake \
  -DENABLE_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=${CXX:-amdclang++} \
  -DCMAKE_HIP_ARCHITECTURES=${HIP_ARCH:-gfx942}\
  -DKokkos_ROOT="${KOKKOS_ROOT}" \
  -DUMPIRE_DIR="${UMPIRE_DIR}" \
  ..
