cmake_minimum_required(VERSION 2.8)
project( pca_rot )
find_package( OpenCV REQUIRED )

set(HEADER_FILES main.cpp
  Object_Boundary.cpp
  RGBD_Image_Processing.cpp
  RGBD_Image_Processing.h
  Object_Boundary.h
  contour_thinning.h)

add_executable( main ${HEADER_FILES} )
target_link_libraries( main ${OpenCV_LIBS} )
