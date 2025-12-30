#include "UmpireSpace.hpp"

#include <Kokkos_Core.hpp>
#include <cassert>
#include <impl/Kokkos_SharedAlloc_timpl.hpp>
#include <iostream>

struct IntegrationTag {};

using IntegrationSpace = UmpireSpace<Kokkos::HostSpace, IntegrationTag>;
using IntegrationView = Kokkos::View<double *, IntegrationSpace>;

KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(IntegrationSpace);
KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(IntegrationSpace);

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    IntegrationSpace::set_allocator("HOST");

    const int N = 10;
    IntegrationView a("A_umpire_integration", N);

    Kokkos::parallel_for(
        "FillIntegrationView", N,
        KOKKOS_LAMBDA(const int i) { a(i) = 1.5 * i; });
    Kokkos::fence();

    Kokkos::View<double *, Kokkos::HostSpace> a_h("A_host_integration", N);
    Kokkos::deep_copy(a_h, a);

    for (int i = 0; i < N; ++i) {
      assert(a_h(i) == 1.5 * i);
    }
  }
  Kokkos::finalize();
  return 0;
}

