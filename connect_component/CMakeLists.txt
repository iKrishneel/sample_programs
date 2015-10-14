cmake_minimum_required(VERSION 2.8)
project( connect_component )
find_package( OpenCV REQUIRED )

set(HEADER_FILES main.cpp
  connectedcomponents.cpp
  precomp.hpp)

add_executable( main ${HEADER_FILES} )
target_link_libraries( main ${OpenCV_LIBS} )
