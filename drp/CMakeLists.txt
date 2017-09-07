cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -Wall)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(utils SHARED
    spscqueue.cc
    mpscqueue.cc
    fex.cc
)

add_executable(drp
    main.cc
)

target_include_directories(utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(drp
    utils
    pthread
    xtc
    pds
)

add_executable(pgp-sender
    pgp-sender.cc
)

add_executable(test-queues
    test.cc
)
target_link_libraries(test-queues
    utils
    pthread
    xtc
)
