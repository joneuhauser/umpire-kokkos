#include "UmpireSpace.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include <cstdio>
#include <impl/Kokkos_SharedAlloc_timpl.hpp>

struct Host {};
struct Pool {};

#ifdef KOKKOS_ENABLE_HIP
using PoolSpaceType = UmpireSpace<Kokkos::HIPSpace, Pool>;
using ViewType = Kokkos::View<double *, PoolSpaceType>;
#else
using PoolSpaceType = UmpireSpace<Kokkos::HostSpace, Pool>;
using ViewType = Kokkos::View<double *, PoolSpaceType>;
#endif
using HostSpaceType = UmpireSpace<Kokkos::HostSpace, Host>;
using HostViewType = Kokkos::View<double *, HostSpaceType>;

KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(HostSpaceType);
KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(PoolSpaceType);

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);

  {
    auto &rm = umpire::ResourceManager::getInstance();
    auto allocator = rm.makeAllocator<umpire::strategy::QuickPool>(
        "pool", rm.getAllocator("HOST"), 1024 * 1024);

    HostSpaceType::set_allocator("HOST");
    PoolSpaceType::set_allocator("pool");

    HostViewType a_h("A_host", 10);
    ViewType a("A", 10);

    Kokkos::parallel_for(10, KOKKOS_LAMBDA(const int i) { a(i) = i; });

    Kokkos::fence();

    Kokkos::deep_copy(a_h, a);

    for (int i{0}; i < 10; ++i) {
      std::cout << "a(i) = " << a_h(i) << "\n";
    }
  }

  Kokkos::finalize();

  return 0;
}

KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(HostSpaceType);
KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(PoolSpaceType);
