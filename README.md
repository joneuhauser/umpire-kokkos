# UmpireSpace Documentation and Quick Start Guide

UmpireSpace is a new Kokkos memory space which utilizes Umpire's memory management capabilities.

## Introduction

[Umpire](https://github.com/llnl/Umpire), developed at LLNL, is an open-source C++ library (with a Fortran interface) that provides a unified, portable 
memory management API to accommodate modern HPC platforms with complex combinations of memory resources. 
Umpire provides support for querying memory resources, provisioning and allocating memory, and memory introspection.
The Umpire [tutorial](https://umpire.readthedocs.io/en/develop/sphinx/tutorial.html) provides a step by step introduction 
to Umpire features.

[Kokkos](https://github.com/kokkos/kokkos), developed at SNL, is a programming model in C++ for writing performance portable applications targeting all major 
HPC platforms. Kokkos is designed to target complex node architectures with N-level memory hierarchies and multiple types 
of execution resources usingB CUDA, HIP, SYCL, HPX, OpenMP and C++ threads as backend programming models.
The Kokkos [Tutorials](https://github.com/kokkos/kokkos-tutorials) are organized within a separate repository under the Kokkos organization on GitHub.

Umpire has a number of different allocation strategies including various memory pools, such as ``QuickPool``, which
have demonstrated substantial performance gains for HPC applications. Although Kokkos provides its own support for 
portable memory management, it does not currently support memory pools, so using UmpireSpace could lead to significant memory performance gains. 

Umpire is built to be modular and can therefore 
be integrated and used with Kokkos. With growing demand to use Umpire and Kokkos together, Umpire established a 
collaboration with Kokkos to create UmpireSpace.

## Build Instructions

Currently UmpireSpace is just a .hpp file. The Umpire team plans to integrate this header into the Umpire repo on Github (see link above). However, until
the header is added in to Umpire, we have the following build instructions so that you can get started today.

First, we will need to clone, build, and install Umpire. Next, we will do the same for Kokkos. Finally, we will clone, build, and run a simple UmpireSpace
example that integrates Umpire and Kokkos together.

Note that these build instructions are specific to LLNL's RzAdams machine which has AMD Instinct MI300A processors and is running TOSS4.

### Build Umpire v2025.12.0

First, build Umpire v2025.12.0 with C++20 support.

Start by cloning the Umpire repo and checking out the v2025.12.0 version.
```
git clone https://github.com/llnl/Umpire.git
git submodule update --init --recursive
git checkout v2025.12.0
```
On RzAdams, I loaded the following submodules:
```
ml rocm/6.4.0
ml cmake/3.24.2
```
Next, build and install Umpire:
```
cmake -DROCM_ROOT_DIR=/opt/rocm-6.4.0 -DHIP_PATH=/opt/rocm-6.4.0/llvm/bin -DCMAKE_CXX_COMPILER=/opt/rocm-6.4.0/bin/hipcc -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_CXX_STANDARD=20 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/g/g0/belcher6/kokkos-stuff/umpire-install -DBLT_CXX_STD=c++20 -DENABLE_HIP=On -DCMAKE_EXE_LINKER_FLAGS="-lpthread" ../

make -j install
```

>[!NOTE]
>For my Umpire build on RzAdams, I needed the `-lpthread` flag in order to build with the `hipcc` compiler.

### Build Kokkos 5.0.0

Now build Kokkos 5.0.0 with C++20. 

Start by cloning the Kokkos repo and checking out the 5.0.0 version.
```
git clone https://github.com/kokkos/kokkos.git
git checkout 5.0.0
```
Build and install Kokkos.

>[!NOTE]
>On RzAdams, I needed to use the `-DKokkos_ARCH_AMD_GFX942_APU=ON` cmake option when building.
>Otherwise I got runtime errors and other warnings such as `Kokkos::HIP::initialize WARNING: running kernels for MI300X (discrete GPU) on a MI300A (APU).`
```
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/g/g0/belcher6/kokkos-stuff/kokkos-install -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_COMPILER=/opt/rocm-6.4.0/bin/hipcc -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX942_APU=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DKokkos_ARCH_ZEN4=On -DCRAYPE_LINK_TYPE=dynamic ../

make -j install
```

### Build UmpireSpace:

Lastly, build UmpireSpace and link in the installations of Kokkos and Umpire that you just built above.

Start by cloning the UmpireSpace repo.
```
git clone https://github.com/llnl/umpire-kokkos.git
mkdir build
cd build
```
Build UmpireSpace:
```
cmake -DENABLE_HIP=On -DCMAKE_BUILD_TYPE=Release -DKokkos_ROOT="/g/g0/belcher6/kokkos-stuff/kokkos-install/lib64/cmake/Kokkos/" -DUMPIRE_DIR="/g/g0/belcher6/kokkos-stuff/umpire-install/lib64/cmake/umpire/" -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_COMPILER=/opt/rocm-6.4.0/bin/hipcc -DCMAKE_HIP_ARCHITECTURES=gfx942 ../

make -j
```

>[!WARNING]
>Because we are still working to get UmpireSpace fully integrated into Kokkos, you will still see this error:
>```
>In file included from /g/g0/belcher6/kokkos-stuff/umpire-kokkos/umpire_space.cpp:4:
>In file included from /g/g0/belcher6/kokkos-stuff/kokkos-install/include/impl/Kokkos_SharedAlloc_timpl.hpp:12:
>/g/g0/belcher6/kokkos-stuff/kokkos-install/include/Kokkos_HostSpace.hpp:6:15: error: static assertion failed: Including non-public Kokkos header files is not allowed.
>    6 | static_assert(false,
>      |               ^~~~~
>1 error generated when compiling for gfx942.
>```
>However, this is easily fixed with:
>```
>vi ../../kokkos-install/include/Kokkos_HostSpace.hpp
>(comment out assert statement on line 6)
>```

After making any edits to the Kokkos install, continue building:
```
make -j
```

### To test UmpireSpace:

Now we can run an example UmpireSpace program.

>[!NOTE]
>In order for this to work, I needed to make sure `HSA_XNACK` was set to 1.

```
export HSA_XNACK=1
./umpire_space
```

```
Expected output:
[belcher6@rzadams1001:build]$ ./umpire_space 
a(i) = 0
a(i) = 1
a(i) = 2
a(i) = 3
a(i) = 4
a(i) = 5
a(i) = 6
a(i) = 7
a(i) = 8
a(i) = 9
```
***Congratulations!*** Now you are able to utilize Umpire for your Kokkos application's memory needs.

## Getting in Contact

I had several issues revolving around the use of a consistent compiler (used `hipcc` for all 3 builds), specifying the correct architecture in the build commands,
and using C++ 20 flags. It took several iterations to get just the right build recipe. Umpire and Kokkos teams will be working to make this process smoother.

For now, if you have trouble building UmpireSpace or any other questions regarding UmpireSpace, reach out to the Umpire team by emailing umpire-dev[at]llnl.gov.
Alternatively, you can also [make a new issue](https://github.com/llnl/umpire-kokkos/issues/new/choose).

# Release

UmpireSpace is released under an MIT license from Umpire. For more details, please see the
[LICENSE](./LICENSE) and [RELEASE](./RELEASE) files.

`LLNL-CODE-747640`
`OCEC-18-031`
