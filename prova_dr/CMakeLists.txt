cmake_minimum_required(VERSION 2.6)

#ds project name
project(dense3Dtracking LANGUAGES C CXX CUDA)

#ds set cmake build flags
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -O3 -DNDEBUG") #ds enable all warnings, c++11 features, enable maximum optimization and disable debug symbols
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

#ds find and set up dependencies
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(CUDA 9.1 REQUIRED)
if(NOT CUDA_FOUND)
  message(STATUS "CUDA not found")
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#ds set internal includes (our files)
include_directories(src)
include_directories(libs)

#ds add build targets
add_subdirectory(executables)
