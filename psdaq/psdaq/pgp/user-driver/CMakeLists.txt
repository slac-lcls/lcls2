cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

add_compile_options(-Wall)
add_compile_options(-Wno-uninitialized)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(pgpdriver
    pgpdriver.cc
)

target_include_directories(pgpdriver PUBLIC
    ../../../../drp)


add_executable(status
    status.cc
)
target_link_libraries(status
    pgpdriver
)

add_executable(read
    read.cc
)
target_link_libraries(read
    pthread
    pgpdriver
)
