cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(psana)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_INSTALL_PREFIX}/lib/cmake/xtcdata")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_INSTALL_PREFIX}/lib/cmake/psalg")

find_package(xtcdata REQUIRED)
find_package(psalg REQUIRED)

