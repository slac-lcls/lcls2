cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(psana)
message("${CMAKE_INSTALL_PREFIX}")
list(APPEND CMAKE_PREFIX_PATH "/sdf/scratch/users/m/mavaylon/lcls2/install/lib/cmake/xtcdata")
list(APPEND CMAKE_PREFIX_PATH "/sdf/scratch/users/m/mavaylon/lcls2/install/lib/cmake/psalg")

find_package(xtcdata REQUIRED)
find_package(psalg REQUIRED)

