cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_compile_options(-Wall)
add_compile_options(-Wno-uninitialized)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(pgpdriver SHARED
    pgpdriver.cc
    aximicronn25q.cc
    mcsfile.cc
)

target_include_directories(pgpdriver PUBLIC
    ../../../drp
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../..>
    $<INSTALL_INTERFACE:include>
)

add_executable(status
    status.cc
)
target_link_libraries(status
    pgpdriver
    rt
)

#add_executable(pgpread
#    pgpread.cc
#)

#target_link_libraries(pgpread
#    pgpdriver
#    xtcdata::xtc
#    Threads::Threads
#    rt
#)

add_executable(pgpmon
    pgpmon.cc
)

target_link_libraries(pgpmon
    pgpdriver
    Threads::Threads
    rt
)

add_executable(looptest
    looptest.cc
)
target_link_libraries(looptest
    pgpdriver
    rt
)

add_executable(tdetsim
    tdetsim.cc)

target_link_libraries(tdetsim
    pgpdriver
    rt
)

add_executable(promload
    promload.cc
)
target_link_libraries(promload
    pgpdriver
    rt
)

add_executable(mcstest
    mcstest.cc
)
target_link_libraries(mcstest
    pgpdriver
    rt
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
