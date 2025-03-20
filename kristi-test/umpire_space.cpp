#include <umpire/ResourceManager.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include <Kokkos_Core.hpp>
#include <cstdio>

template <typename MemorySpace = Kokkos::HostSpace> class UmpireSpace {
public:
  using memory_space = UmpireSpace<MemorySpace>;
  using size_type = size_t;
  using umpire_space = UmpireSpace<MemorySpace>;
  using execution_space = typename MemorySpace::execution_space;
  using device_type = Kokkos::Device<execution_space, MemorySpace>;

  explicit UmpireSpace(const std::string &allocator_name)
      : m_allocator_name(allocator_name) {}

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

  //static constexpr bool is_host_accessible_space() {
  //  return Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
  //                                         MemorySpace>::accessible;
  //}

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
} // namespace Impl

} // namespace Kokkos

//----------------------------------------------------------------------------

//KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(UmpireSpace<Kokkos::HostSpace>);
//KOKKOS_IMPL_SHARED_ALLOCATION_RECORD_EXPLICIT_INSTANTIATION(
//    UmpireSpace<Kokkos::HostSpace>);
//#include <impl/Kokkos_SharedAlloc_timpl.hpp>

namespace Kokkos {
namespace Impl {

template <>
class SharedAllocationRecord<UmpireSpace<Kokkos::HostSpace>, void>
    : public SharedAllocationRecordCommon<UmpireSpace<Kokkos::HostSpace>> {
public:
  using RecordBase = SharedAllocationRecordCommon<UmpireSpace<Kokkos::HostSpace>>;

  SharedAllocationRecord(
      const UmpireSpace<Kokkos::HostSpace>& space,
      const std::string& label,
      const size_t alloc_size,
      function_type dealloc_func = &RecordBase::deallocate)
      : RecordBase(space, label, alloc_size, dealloc_func) {}

  static SharedAllocationRecord* allocate(
      const UmpireSpace<Kokkos::HostSpace>& space,
      const std::string& label,
      const size_t alloc_size) {
    return new SharedAllocationRecord(space, label, alloc_size);
  }

  void* data() const {
    return reinterpret_cast<void*>(RecordBase::m_alloc_ptr + sizeof(SharedAllocationHeader));
  }

  static void deallocate(SharedAllocationRecord<void, void>* record) {
    delete static_cast<SharedAllocationRecord*>(record);
  }
};

} // namespace Impl
} // namespace Kokkos


//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

// Add these implementations before the DeepCopy specializations

inline void host_to_umpire_deep_copy(void *dst, const void *src, size_t n,
                                     bool = true) {
  std::memcpy(dst, src, n);
}

inline void umpire_to_host_deep_copy(void *dst, const void *src, size_t n,
                                     bool = true) {
  std::memcpy(dst, src, n);
}

inline void umpire_to_umpire_deep_copy(void *dst, const void *src, size_t n) {
  std::memcpy(dst, src, n);
}

template <typename MemorySpace>
void *checked_allocation_with_header(const MemorySpace &space,
                                     const std::string &label,
                                     size_t alloc_size) {
  void *ptr = space.allocate(sizeof(SharedAllocationHeader) + alloc_size);
  if (!ptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

} // namespace Impl
} // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <class ExecutionSpace>
struct DeepCopy<UmpireSpace<Kokkos::HostSpace>, Kokkos::HostSpace,
                ExecutionSpace> {
  DeepCopy(void *dst, const void *src, size_t n) {
    host_to_umpire_deep_copy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n) {
    exec.fence();
    host_to_umpire_deep_copy(dst, src, n);
    exec.fence();
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::HostSpace, UmpireSpace<Kokkos::HostSpace>,
                ExecutionSpace> {
  DeepCopy(void *dst, const void *src, size_t n) {
    umpire_to_host_deep_copy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n) {
    exec.fence();
    umpire_to_host_deep_copy(dst, src, n);
    exec.fence();
  }
};

template <class ExecutionSpace>
struct DeepCopy<UmpireSpace<Kokkos::HostSpace>, UmpireSpace<Kokkos::HostSpace>,
                ExecutionSpace> {
  DeepCopy(void *dst, const void *src, size_t n) {
    umpire_to_umpire_deep_copy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n) {
    exec.fence();
    umpire_to_umpire_deep_copy(dst, src, n);
    exec.fence();
  }
};

#ifdef KOKKOS_ENABLE_CUDA
template <class ExecutionSpace>
struct DeepCopy<UmpireSpace<Kokkos::CudaSpace>, Kokkos::HostSpace,
                ExecutionSpace> {
  DeepCopy(void *dst, const void *src, size_t n) {
    host_to_umpire_deep_copy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n) {
    exec.fence();
    host_to_umpire_deep_copy(dst, src, n);
    exec.fence();
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::HostSpace, UmpireSpace<Kokkos::CudaSpace>,
                ExecutionSpace> {
  DeepCopy(void *dst, const void *src, size_t n) {
    umpire_to_host_deep_copy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n) {
    exec.fence();
    umpire_to_host_deep_copy(dst, src, n);
    exec.fence();
  }
};

template <class ExecutionSpace>
struct DeepCopy<UmpireSpace<Kokkos::CudaSpace>, UmpireSpace<Kokkos::CudaSpace>,
                ExecutionSpace> {
  DeepCopy(void *dst, const void *src, size_t n) {
    umpire_to_umpire_deep_copy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n) {
    exec.fence();
    umpire_to_umpire_deep_copy(dst, src, n);
    exec.fence();
  }
};
#endif // KOKKOS_ENABLE_CUDA

} // namespace Impl
} // namespace Kokkos

using ViewType = Kokkos::View<double *, UmpireSpace<>>;

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);

  {
    auto &rm = umpire::ResourceManager::getInstance();
    auto allocator = rm.makeAllocator<umpire::strategy::QuickPool>(
        "pool", rm.getAllocator("HOST"), 1024 * 1024);
    UmpireSpace<> umpire_space("pool");
    ViewType a(Kokkos::view_alloc(umpire_space, "A"), 10);

    Kokkos::parallel_for(
        10, KOKKOS_LAMBDA(const int i) { a(i) = i; });

    Kokkos::parallel_for(
        10,
        KOKKOS_LAMBDA(const int i) { printf("a(%d) = %f\n", i,  a(i)); });
  }

  Kokkos::finalize();

  return 0;
}
