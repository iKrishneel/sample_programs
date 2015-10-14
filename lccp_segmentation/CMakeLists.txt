cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(lccp_segmentation)
find_package(PCL 1.7 REQUIRED)
include_directories ("${PROJECT_SOURCE_DIR}")

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(HEADER_FILES main.cpp 
		lccp_segmentation.cpp
		${PROJECT_SOURCE_DIR}/lccp_segmentation.h)

add_executable (main main.cpp)
target_link_libraries (main ${PCL_LIBRARIES})
