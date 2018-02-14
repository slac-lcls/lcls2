
add_library(bld SHARED
    HpsEventIterator.cc
    Server.cc
    Client.cc
)

target_include_directories(bld PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)

install(TARGETS bld
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
