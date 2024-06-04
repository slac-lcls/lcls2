cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(psana)
# cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/sdf/scratch/users/m/mavaylon/lcls2/install 

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(xtcdata REQUIRED)
find_package(psalg REQUIRED)

