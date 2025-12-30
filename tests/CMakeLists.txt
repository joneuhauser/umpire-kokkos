add_executable(test_basic_allocation test_basic_allocation.cpp)
add_executable(test_deep_copy test_deep_copy.cpp)
add_executable(test_pool_allocator test_pool_allocator.cpp)
add_executable(test_traits test_traits.cpp)
add_executable(test_integration test_integration.cpp)

target_include_directories(test_basic_allocation PRIVATE 
  ${CMAKE_SOURCE_DIR}
)

target_include_directories(test_deep_copy PRIVATE 
  ${CMAKE_SOURCE_DIR}
)

target_include_directories(test_pool_allocator PRIVATE 
  ${CMAKE_SOURCE_DIR}
)

target_include_directories(test_traits PRIVATE 
  ${CMAKE_SOURCE_DIR}
)

target_include_directories(test_integration PRIVATE 
  ${CMAKE_SOURCE_DIR}
)

target_link_libraries(test_basic_allocation PRIVATE
  Kokkos::kokkos
  umpire
)

target_link_libraries(test_deep_copy PRIVATE
  Kokkos::kokkos
  umpire
)

target_link_libraries(test_pool_allocator PRIVATE
  Kokkos::kokkos
  umpire
)

target_link_libraries(test_traits PRIVATE
  Kokkos::kokkos
  umpire
)

target_link_libraries(test_integration PRIVATE
  Kokkos::kokkos
  umpire
)

add_test(NAME test_basic_allocation COMMAND test_basic_allocation)
add_test(NAME test_deep_copy COMMAND test_deep_copy)
add_test(NAME test_pool_allocator COMMAND test_pool_allocator)
add_test(NAME test_traits COMMAND test_traits)
add_test(NAME test_integration COMMAND test_integration)

