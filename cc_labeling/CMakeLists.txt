cmake_minimum_required(VERSION 2.8)
project( cc_labeling )
find_package( OpenCV REQUIRED )

include_directories (
  "${PROJECT_SOURCE_DIR}"/connected.h
  "${PROJECT_SOURCE_DIR}"/contour_thinning.h)

add_executable(
  main main.cpp
  #connected.cpp
  )

target_link_libraries( main ${OpenCV_LIBS} ${PROJECT_SOURCE_DIR} )
