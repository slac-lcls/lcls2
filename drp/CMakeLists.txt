cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(utils SHARED
    mpscqueue.cc
    fex.cc
)

add_executable(drp
    main.cc
)

target_include_directories(utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(drp
    utils
    pthread
    xtc
)

add_executable(pgp-sender
    pgp-sender.cc
)

add_executable(test-pgp-read
    test-pgp-read.cc
)
#target_link_libraries(test-queues
#    utils
#    pthread
#    xtc
#)
