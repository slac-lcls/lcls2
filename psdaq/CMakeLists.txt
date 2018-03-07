cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

add_compile_options(-Wall)
add_compile_options(-Wno-uninitialized)
add_compile_options(-fno-omit-frame-pointer)

find_package(xtcdata REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(psdaq)
add_subdirectory(drp)
