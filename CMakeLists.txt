cmake_minimum_required(VERSION 3.2 FATAL_ERROR)


find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs 3.5 REQUIRED)

set(CMAKE_BUILD_TYPE Debug)

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
    "import numpy; print(numpy.get_include(), end='')"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HDF5_ROOT /reg/neh/home4/weninc/.conda/envs/lcls2)

find_package(HDF5 REQUIRED HL)

add_subdirectory(pdsdata)
add_subdirectory(xtchdf5)
add_subdirectory(drp)
