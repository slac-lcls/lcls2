cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(psdaq)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_compile_options(-Wall)
add_compile_options(-Wno-uninitialized)
add_compile_options(-fno-omit-frame-pointer)

find_package(xtcdata REQUIRED)
find_package(psalg REQUIRED)
find_package(libfabric REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(prometheus-cpp REQUIRED)
find_package(ZeroMQ REQUIRED)
find_package(epics REQUIRED)
find_package(cpsw REQUIRED)
find_package(RapidJSON REQUIRED)

find_library(YAML_CPP_LIB yaml-cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# This is a temporary fix to pick up the new lib libstdc++ from conda
link_directories(${CMAKE_PREFIX_PATH}/lib)

add_subdirectory(psdaq)
add_subdirectory(drp)
add_subdirectory(epicsArch)
