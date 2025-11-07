cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.27.0")
  cmake_policy(SET CMP0148 OLD)
endif()

project(psdaq)

include(FindCUDAToolkit)
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  # Ensure nvcc uses the host compiler from our environment
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin $ENV{CC}")
  #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin $ENV{CC} -rdc=true")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_compile_options(-Wall)
add_compile_options(-Wno-uninitialized)
add_compile_options(-fno-omit-frame-pointer)

find_package(libfabric REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(prometheus-cpp REQUIRED)
find_package(ZeroMQ REQUIRED)
find_package(epics REQUIRED)
find_package(RapidJSON REQUIRED)
find_package(Readline REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(slsDetectorPackage REQUIRED)

find_library(YAML_CPP_LIB yaml-cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(psdaq)
add_subdirectory(drp)
add_subdirectory(epicsArch)
add_subdirectory(drpGpu)
