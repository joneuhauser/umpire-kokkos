add_executable(tutorial_umpire_host tutorial_umpire_host.cpp)

target_include_directories(tutorial_umpire_host PRIVATE 
  ${CMAKE_SOURCE_DIR}
)

target_link_libraries(tutorial_umpire_host PRIVATE
  Kokkos::kokkos
  umpire
)

