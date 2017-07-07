cmake_minimum_required(VERSION 3.2 FATAL_ERROR)


find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HDF5_ROOT /reg/neh/home4/weninc/.conda/envs/lcls2)

find_package(HDF5 REQUIRED HL)

add_subdirectory(pdsdata)
