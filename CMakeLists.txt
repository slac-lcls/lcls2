cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(lcls2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall)
add_compile_options(-Wno-uninitialized)
add_compile_options(-fno-omit-frame-pointer)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_subdirectory(xtcdata)
add_subdirectory(psalg)
add_subdirectory(psana)
