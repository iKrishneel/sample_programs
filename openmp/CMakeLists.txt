cmake_minimum_required(VERSION 2.8)
project(openmp)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(hello_world hello_world.cpp )
target_link_libraries( hello_world ${OPENMP_LIBS} )

add_executable(for_loop for_loop.cpp )
target_link_libraries( for_loop ${OPENMP_LIBS})

add_executable(section section.cpp )
target_link_libraries( section ${OPENMP_LIBS})

add_executable(work_sharing_single work_sharing_single.cpp)
target_link_libraries( work_sharing_single ${OPENMP_LIBS})

add_executable(critial critial.cpp)
target_link_libraries( critial ${OPENMP_LIBS})

add_executable(barrier barrier.cpp)
target_link_libraries( barrier ${OPENMP_LIBS})

add_executable(atomic atomic.cpp)
target_link_libraries( atomic ${OPENMP_LIBS})