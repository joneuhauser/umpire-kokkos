#include "UmpireSpace.hpp"

#include <Kokkos_Core.hpp>
#include <cassert>
#include <impl/Kokkos_SharedAlloc_timpl.hpp>

struct BasicTag {};

using BasicSpace = UmpireSpace<Kokkos::HostSpace, BasicTag>;
using BasicView = Kokkos::View<int *, BasicSpace>;

KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(BasicSpace);
KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(BasicSpace);

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    BasicSpace::set_allocator("HOST");

    const int N = 16;
    BasicView v("basic_view", N);

    Kokkos::parallel_for(
        "FillBasicView", N, KOKKOS_LAMBDA(const int i) { v(i) = 42; });
    Kokkos::fence();

    Kokkos::View<int *, Kokkos::HostSpace> v_h("basic_view_host", N);
    Kokkos::deep_copy(v_h, v);

    for (int i = 0; i < N; ++i) {
      assert(v_h(i) == 42);
    }
  }
  Kokkos::finalize();
  return 0;
}

