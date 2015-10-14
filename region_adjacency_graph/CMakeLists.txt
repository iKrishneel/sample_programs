cmake_minimum_required(VERSION 2.8)
project( region_grow )
find_package( OpenCV REQUIRED)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

add_executable( main main.cpp )
target_link_libraries( main ${OpenCV_LIBS} ${Boost_LIBRARIES})
