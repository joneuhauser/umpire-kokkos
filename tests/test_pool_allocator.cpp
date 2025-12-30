#include "UmpireSpace.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

#include <Kokkos_Core.hpp>
#include <cassert>
#include <impl/Kokkos_SharedAlloc_timpl.hpp>

struct PoolTag {};

using PoolSpace = UmpireSpace<Kokkos::HostSpace, PoolTag>;
using PoolView = Kokkos::View<double *, PoolSpace>;

KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(PoolSpace);
KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(PoolSpace);

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    auto &rm = umpire::ResourceManager::getInstance();
    auto host_alloc = rm.getAllocator("HOST");

    auto pool_alloc = rm.makeAllocator<umpire::strategy::QuickPool>(
        "test_pool", host_alloc, 1024 * 1024);

    (void)pool_alloc;

    PoolSpace::set_allocator("test_pool");

    const int N = 64;
    PoolView v1("pool_view1", N);
    PoolView v2("pool_view2", N);

    Kokkos::parallel_for(
        "FillPoolViews", N, KOKKOS_LAMBDA(const int i) {
          v1(i) = static_cast<double>(i);
          v2(i) = static_cast<double>(2 * i);
        });
    Kokkos::fence();

    Kokkos::View<double *, Kokkos::HostSpace> h1("h1_pool", N);
    Kokkos::View<double *, Kokkos::HostSpace> h2("h2_pool", N);

    Kokkos::deep_copy(h1, v1);
    Kokkos::deep_copy(h2, v2);

    for (int i = 0; i < N; ++i) {
      assert(h1(i) == static_cast<double>(i));
      assert(h2(i) == static_cast<double>(2 * i));
    }
  }
  Kokkos::finalize();
  return 0;
}

