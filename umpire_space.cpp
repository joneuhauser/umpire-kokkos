#include "UmpireSpace.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include <cstdio>
#include <impl/Kokkos_SharedAlloc_timpl.hpp>

KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    UmpireSpace<Kokkos::HostSpace>);
KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
    UmpireSpace<Kokkos::HIPSpace>);

using ViewType = Kokkos::View<double *, UmpireSpace<Kokkos::HIPSpace>>;
using HostViewType = Kokkos::View<double *, UmpireSpace<Kokkos::HostSpace>>;

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);

  {
    auto &rm = umpire::ResourceManager::getInstance();
    auto allocator = rm.makeAllocator<umpire::strategy::QuickPool>(
        "pool", rm.getAllocator("DEVICE"), 1024 * 1024);

    UmpireSpace<> umpire_host_space("HOST");
    HostViewType a_h(Kokkos::view_alloc(umpire_host_space, "A_host"), 10);

    UmpireSpace<Kokkos::HIPSpace> umpire_space("pool");
    ViewType a(Kokkos::view_alloc(umpire_space, "A"), 10);

    Kokkos::parallel_for(
        10, KOKKOS_LAMBDA(const int i) { a(i) = i; });

    Kokkos::fence();

    Kokkos::deep_copy(a_h, a);

    for(int i{0}; i < 10; ++i) {
      std::cout << "a(i) = " << a_h(i) << "\n";
    }
  }

  Kokkos::finalize();

  return 0;
}
