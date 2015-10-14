cmake_minimum_required(VERSION 2.8)
project( optical_flow )
find_package( OpenCV REQUIRED )

add_executable( main main.cpp )
add_executable( fback fback.cpp )

target_link_libraries( main ${OpenCV_LIBS} )
target_link_libraries( fback ${OpenCV_LIBS} )
