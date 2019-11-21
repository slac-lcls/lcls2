cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(xtc_vol)

set(HDF5_HOME "/Users/tonglin/nersc_dev_sync/hdf5_build/hdf5")
include_directories(${HDF5_HOME}/include .)
link_directories(${HDF5_HOME}/lib .)
add_library(h5xtc xtc_vol.c)

add_executable(testcase_xtc_vol testcase_xtc_vol.c)
target_link_libraries(testcase_xtc_vol h5xtc hdf5 mpi z)
