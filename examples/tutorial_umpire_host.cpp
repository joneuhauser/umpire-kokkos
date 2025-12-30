#include "UmpireSpace.hpp"

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_SharedAlloc_timpl.hpp>
#include <iostream>

struct HostTag {};

using HostUmpireSpace = UmpireSpace<Kokkos::HostSpace, HostTag>;
using ViewType = Kokkos::View<double *, HostUmpireSpace>;

KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(HostUmpireSpace);

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    auto &rm = umpire::ResourceManager::getInstance();

    // Use the HOST allocator directly for this simple example
    HostUmpireSpace::set_allocator("HOST");

    const int N = 10;
    ViewType a("A_umpire_host", N);

    Kokkos::parallel_for(
        "FillView", N, KOKKOS_LAMBDA(const int i) { a(i) = 1.5 * i; });

    Kokkos::fence();

    // Mirror to a standard HostSpace view for printing
    Kokkos::View<double *, Kokkos::HostSpace> a_h("A_host_mirror", N);
    Kokkos::deep_copy(a_h, a);

    std::cout << "Values in A_umpire_host:\n";
    for (int i = 0; i < N; ++i) {
      std::cout << "A(" << i << ") = " << a_h(i) << "\n";
    }
  }
  Kokkos::finalize();
  return 0;
}

KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(HostUmpireSpace);
