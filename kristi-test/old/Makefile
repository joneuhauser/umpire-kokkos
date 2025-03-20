# Compiler and architecture settings
CXX = /opt/rocm-6.3.0/llvm/bin/amdclang++
HIP_ARCH = gfx942

# Paths to dependencies
KOKKOS_PATH = /usr/workspace/belcher6/kokkos-install
UMPIRE_PATH = /usr/workspace/belcher6/umpire-install

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Werror -fPIE --rocm-path=/opt/rocm-6.3.0 -x hip --offload-arch=${HIP_ARCH} -Wno-unused-result -Wno-unused-parameter -Wno-unused-variable
INCLUDES = -I$(KOKKOS_PATH)/include -I$(UMPIRE_PATH)/include -isystem /opt/rocm-6.3.0/include
LIBS = -L$(KOKKOS_PATH)/lib64 -L$(UMPIRE_PATH)/lib64 -L/opt/rocm-6.3.0/lib
LDFLAGS = -lkokkoscore -lumpire -ldl 

#/opt/rocm-6.3.0/lib/llvm/lib/clang/18/lib/linux/libclang_rt.builtins-x86_64.a

# Target executable
TARGET = umpire_space_test

# Source files
SRCS = umpire_space.cpp

# Rules
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET)
