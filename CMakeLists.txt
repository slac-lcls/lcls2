cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(lcls2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall)
add_compile_options(-Wno-uninitialized)
add_compile_options(-fno-omit-frame-pointer)

add_subdirectory(xtcdata)
add_subdirectory(psalg)
add_subdirectory(psana)

if(NOT NO_DAQ)
    message(STATUS "DAQ INCLUDED")
    add_subdirectory(psdaq)
endif()