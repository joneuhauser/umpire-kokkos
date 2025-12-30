#include "UmpireSpace.hpp"

#include <Kokkos_Core.hpp>

struct TraitsTag {};

using TraitsSpace = UmpireSpace<Kokkos::HostSpace, TraitsTag>;

int main(int, char **) {
  static_assert(TraitsSpace::is_host_accessible_space(),
                "UmpireSpace wrapping HostSpace must be host accessible");

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace, TraitsSpace>::accessible,
      "HostSpace must be able to access UmpireSpace<HostSpace>");

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<TraitsSpace, Kokkos::HostSpace>::accessible,
      "UmpireSpace<HostSpace> must be able to access HostSpace");

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<TraitsSpace, TraitsSpace>::deepcopy,
      "Deep copy within UmpireSpace<HostSpace> must be allowed");

  return 0;
}

