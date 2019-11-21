cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(xtc_vol)

set(HDF5_HOME "YOUR_HDF5_HOME_DIR")
include_directories(${HDF5_HOME}/include .)
link_directories(${HDF5_HOME}/lib .)
add_library(h5xtc xtc_vol.c)

add_executable(testcase_xtc_vol testcase_xtc_vol.c)
target_link_libraries(testcase_xtc_vol h5xtc hdf5 mpi z)

add_executable(xtc_io_test	xtc_io_api.cc)
target_link_libraries(xtc_io_test	xtc)