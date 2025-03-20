#ifndef KOKKOS_UMPIRE_SPACE_HPP
#define KOKKOS_UMPIRE_SPACE_HPP

#include "umpire/ResourceManager.hpp"
#include <Kokkos_Core.hpp>
#include <string>

template <typename MemorySpace = Kokkos::HostSpace> class UmpireSpace {
public:
  //! Tag this class as a kokkos memory space
  using memory_space = UmpireSpace<MemorySpace>;
  using size_type = size_t;
  using umpire_space = UmpireSpace<MemorySpace>;

  /// \typedef execution_space
  /// \brief Default execution space for this memory space.
  ///
  /// Every memory space has a default execution space.  This is
  /// useful for things like initializing a View (which happens in
  /// parallel using the View's default execution space).
  using execution_space = typename MemorySpace::execution_space;

  //! This memory space preferred device_type
  using device_type = Kokkos::Device<execution_space, MemorySpace>;

  //! Constructor taking the name of the Umpire::Allocator
  explicit UmpireSpace(const std::string &allocator_name)
      : m_allocator_name(allocator_name) {}

  //! Defaulted constructors and assignment operators
  UmpireSpace() = default;
  UmpireSpace(UmpireSpace &&rhs) = default;
  UmpireSpace(const UmpireSpace &rhs) = default;
  UmpireSpace &operator=(UmpireSpace &&) = default;
  UmpireSpace &operator=(const UmpireSpace &) = default;
  ~UmpireSpace() = default;

  //! Allocate memory using the Umpire::Allocator
  void *allocate(size_t size) const {
    return umpire::ResourceManager::getInstance()
        .getAllocator(m_allocator_name)
        .allocate(size);
  }

  template <typename ExecutionSpace>
  void *allocate(const ExecutionSpace &, const char *arg_label,
                 const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const {
    return allocate(arg_alloc_size);
  }

  void *allocate(const char *arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const {
    return allocate(arg_alloc_size);
  }

  //! Deallocate memory using the Umpire::Allocator
  void deallocate(void *ptr, size_t size) const {
    umpire::ResourceManager::getInstance()
        .getAllocator(m_allocator_name)
        .deallocate(ptr);
  }

  void deallocate(const char *arg_label, void *const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) {
    deallocate(arg_alloc_ptr, arg_alloc_size);
  }

  /**\brief Return Name of the MemorySpace */
  static constexpr const char *name() { return "UMPIRE"; }

  static constexpr bool is_host_accessible_space() {
    return Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                           MemorySpace>::accessible;
  }

private:
  const std::string m_allocator_name;
  friend class Kokkos::Impl::SharedAllocationRecord<UmpireSpace<MemorySpace>,
                                                    void>;
};

namespace Kokkos {

namespace Impl {

template <>
struct MemorySpaceAccess<Kokkos::HostSpace, UmpireSpace<Kokkos::HostSpace>> {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<UmpireSpace<Kokkos::HostSpace>, Kokkos::HostSpace> {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy = true };
};

#ifdef KOKKOS_ENABLE_CUDA
template <>
struct MemorySpaceAccess<Kokkos::CudaSpace, UmpireSpace<Kokkos::HostSpace>> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<UmpireSpace<Kokkos::HostSpace>, Kokkos::CudaSpace> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::HostSpace, UmpireSpace<Kokkos::CudaSpace>> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<UmpireSpace<Kokkos::CudaSpace>, Kokkos::HostSpace> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::CudaSpace, UmpireSpace<Kokkos::CudaSpace>> {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<UmpireSpace<Kokkos::CudaSpace>, Kokkos::CudaSpace> {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy = true };
};
#endif // KOKKOS_ENABLE_CUDA

#ifdef KOKKOS_ENABLE_HIP
template <>
struct MemorySpaceAccess<Kokkos::HIPSpace, UmpireSpace<Kokkos::HostSpace>> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<UmpireSpace<Kokkos::HostSpace>, Kokkos::HIPSpace> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::HostSpace, UmpireSpace<Kokkos::HIPSpace>> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<UmpireSpace<Kokkos::HIPSpace>, Kokkos::HostSpace> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::HIPSpace, UmpireSpace<Kokkos::HIPSpace>> {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<UmpireSpace<Kokkos::HIPSpace>, Kokkos::HIPSpace> {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy = true };
};
#endif // KOKKOS_ENABLE_HIP

template <class MemorySpace1, class MemorySpace2, class ExecutionSpace>
struct DeepCopy<UmpireSpace<MemorySpace1>, MemorySpace2, ExecutionSpace> {
  inline DeepCopy(void *dst, const void *src, size_t n) {
    DeepCopy<MemorySpace1, MemorySpace2, ExecutionSpace>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n) {
    DeepCopy<MemorySpace1, MemorySpace2, ExecutionSpace>(exec, dst, src, n);
  }
};

template <class MemorySpace1, class MemorySpace2, class ExecutionSpace>
struct DeepCopy<UmpireSpace<MemorySpace1>, UmpireSpace<MemorySpace2>, ExecutionSpace> {
  inline DeepCopy(void *dst, const void *src, size_t n) {
    DeepCopy<MemorySpace1, MemorySpace2, ExecutionSpace>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n) {
    DeepCopy<MemorySpace1, MemorySpace2, ExecutionSpace>(exec, dst, src, n);
  }
};

template <class MemorySpace1, class MemorySpace2, class ExecutionSpace>
struct DeepCopy<MemorySpace1, UmpireSpace<MemorySpace2>, ExecutionSpace> {
  inline DeepCopy(void *dst, const void *src, size_t n) {
    DeepCopy<MemorySpace1, MemorySpace2, ExecutionSpace>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n) {
    DeepCopy<MemorySpace1, MemorySpace2, ExecutionSpace>(exec, dst, src, n);
  }
};

KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(UmpireSpace<Kokkos::HostSpace>);
#ifdef KOKKOS_ENABLE_HIP
KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(UmpireSpace<Kokkos::HIPSpace>);
#endif
#ifdef KOKKOS_ENABLE_CUDA
KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(UmpireSpace<Kokkos::CudaSpace>);
#endif

} // namespace Impl
} // namespace Kokkos

#endif // KOKKOS_UMPIRE_SPACE_HPP
