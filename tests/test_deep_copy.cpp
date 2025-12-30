#include "UmpireSpace.hpp"

#include <Kokkos_Core.hpp>
#include <cassert>
#include <impl/Kokkos_SharedAlloc_timpl.hpp>

struct DeepCopyTag {};

using DeepCopySpace = UmpireSpace<Kokkos::HostSpace, DeepCopyTag>;
using DeepCopyView = Kokkos::View<double *, DeepCopySpace>;

KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(DeepCopySpace);
KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(DeepCopySpace);

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    DeepCopySpace::set_allocator("HOST");

    const int N = 32;

    DeepCopyView a("a_umpire", N);
    Kokkos::View<double *, Kokkos::HostSpace> h1("h1", N);
    Kokkos::View<double *, Kokkos::HostSpace> h2("h2", N);

    for (int i = 0; i < N; ++i) {
      h1(i) = 1.5 * i;
    }

    Kokkos::deep_copy(a, h1);
    Kokkos::deep_copy(h2, a);

    for (int i = 0; i < N; ++i) {
      assert(h2(i) == h1(i));
    }
  }
  Kokkos::finalize();
  return 0;
}

