# Minimum CMake version
cmake_minimum_required(VERSION 3.18)

# Project name and language
project(UmpireSpaceTest LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable HIP support
set(CMAKE_HIP_ARCHITECTURES gfx942) # Set your GPU architecture here
enable_language(HIP)

# Paths to dependencies
set(Kokkos_DIR /usr/workspace/belcher6/kokkos-install/lib64/cmake/Kokkos)
set(Umpire_DIR /usr/workspace/belcher6/umpire-install/lib64/cmake/umpire)

# Find Kokkos and Umpire
find_package(Kokkos REQUIRED)
find_package(Umpire REQUIRED)

# Include directories
include_directories(${Kokkos_INCLUDE_DIRS} ${Umpire_INCLUDE_DIRS})

# Add the executable
add_executable(umpire_space_test umpire_space.cpp)

# Link libraries
target_link_libraries(umpire_space_test PRIVATE Kokkos::kokkoscore umpire)

#find_package(umpire REQUIRED PATHS /usr/workspace/belcher6/umpire-install)   
#kokkos_add_executable(tutorial_04_simple_memoryspaces SOURCES simple_memoryspaces.cpp)   
#target_link_libraries(Kokkos_tutorial_04_simple_memoryspaces PRIVATE umpire)
