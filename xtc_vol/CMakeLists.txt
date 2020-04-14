# Compiling flags used for debugging purpose, optional.
SET(MY_CC_FLAGS "-g")
set(CMAKE_CXX_FLAGS_DEBUG  "-g")
SET(CMAKE_CXX_FLAGS  "${MY_CC_FLAGS} ${CMAKE_CXX_FLAGS}")

# Project: xtc_vol
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(xtc_vol)

# Dependency: MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM /usr/local/include)#for MPI

# Dependency: XTC
find_library(XTC_LIB xtc HINTS /usr/local)

# Dependecy: HDF5
set(HDF5_HOME "../../hdf5_build/hdf5")
include_directories(/./usr/local/include) # For OSX 10.5+
include_directories(${HDF5_HOME}/include .)
link_directories(${HDF5_HOME}/lib .)

# XTC API in C
add_library(xtc_clib SHARED xtc_io_api.cc xtc_tree.cc)
target_link_libraries(xtc_clib ${XTC_LIB})

# Testcase for tree operations.
add_executable(test_tree tree_test.cc)
target_link_libraries(test_tree xtc_clib)

# Core XTC_VOL library
add_library(h5xtc SHARED xtc_vol.c)
target_link_libraries(h5xtc hdf5 z MPI::MPI_C xtc_clib)
