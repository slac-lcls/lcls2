cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(psana)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(xtcdata REQUIRED)
find_package(psalg REQUIRED)

