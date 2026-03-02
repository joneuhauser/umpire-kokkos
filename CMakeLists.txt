cmake_minimum_required(VERSION 3.18)
project(KokkosUmpire LANGUAGES CXX)

# Option to enable CUDA support
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_HIP "Enable HIP support" OFF)

# Set CMake policies
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable CUDA if requested
if(ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_compile_definitions(KOKKOS_ENABLE_CUDA)
  
  # Set CUDA architecture target based on detected GPU
  include(CheckLanguage)
  check_language(CUDA)
  
  # Set default CUDA architectures if not specified
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86)
  endif()
  
  # CUDA-specific compile options
  set(CUDA_OPTIONS
    -DKOKKOS_ENABLE_CUDA_UVM
    -DKOKKOS_ENABLE_IMPL_CUDA_UNIFIED_MEMORY
  )
endif()

if(ENABLE_HIP)
  enable_language(HIP)
  add_compile_definitions(KOKKOS_ENABLE_HIP)
endif()

message(STATUS "Looking for Kokkos in: ${Kokkos_ROOT}")
find_package(Kokkos REQUIRED CONFIG PATHS "${Kokkos_ROOT}" NO_DEFAULT_PATH)

message(STATUS "Looking for Umpire in: ${UMPIRE_DIR}")
find_package(umpire REQUIRED CONFIG PATHS "${UMPIRE_DIR}" NO_DEFAULT_PATH)

add_executable(umpire_space umpire_space.cpp)

# Set include directories
target_include_directories(umpire_space PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries - now simplified since both packages must be found via find_package
message(STATUS "Linking with Kokkos::kokkos and ${UMPIRE_TARGET}")
target_link_libraries(umpire_space PRIVATE
  Kokkos::kokkos
  umpire
)

# Add CUDA-specific options if enabled
if(ENABLE_CUDA)
  target_compile_definitions(umpire_space PRIVATE ${CUDA_OPTIONS})
  target_link_libraries(umpire_space PRIVATE CUDA::cudart)
  
  # Set the CUDA source properties if using .cu files
  set_source_files_properties(umpire_space.cpp PROPERTIES LANGUAGE CUDA)
endif()

if(ENABLE_HIP)
  set_source_files_properties(umpire_space.cpp PROPERTIES LANGUAGE HIP)
endif()

# Install target
install(TARGETS umpire_space
  RUNTIME DESTINATION bin
)
