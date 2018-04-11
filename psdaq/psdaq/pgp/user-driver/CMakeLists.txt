cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

add_compile_options(-Wall)
add_compile_options(-Wno-uninitialized)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(pgpdriver SHARED
    pgpdriver.cc
)

target_include_directories(pgpdriver PUBLIC
    ../../../drp
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

add_executable(status
    status.cc
)
target_link_libraries(status
    pgpdriver
)

add_executable(read
    read.cc
)
target_include_directories(read PRIVATE
    $ENV{CONDA_PREFIX}/include
)
target_link_libraries(read
    pthread
    pgpdriver
    $ENV{CONDA_PREFIX}/lib/libzmq.so
    $ENV{CONDA_PREFIX}/lib/libstdc++.so.6
)

add_executable(looptest
    looptest.cc
)
target_link_libraries(looptest
    pgpdriver
)

install(FILES
    pgpdriver.h
    DESTINATION include
)

install(TARGETS pgpdriver
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
