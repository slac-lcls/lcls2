cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(psana)
# cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/sdf/scratch/users/m/mavaylon/lcls2/install

find_package(xtcdata REQUIRED)
find_package(psalg REQUIRED)

