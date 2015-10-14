
cmake_minimum_required(VERSION 2.8)
project( libsvm )
find_package( OpenCV REQUIRED )
include_directories ("${PROJECT_SOURCE_DIR}")

set(HEADER_FILES main.cpp
  svm.cpp
  ${PROJECT_SOURCE_DIR}/svm.h)

add_executable(main ${HEADER_FILES})
target_link_libraries( main ${OpenCV_LIBS} )