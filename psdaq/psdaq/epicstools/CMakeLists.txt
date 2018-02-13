
add_library(epicstools SHARED
    EpicsCA.cc
    PvServer.cc
)

target_include_directories(epicstools PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/epics/include
    $ENV{CONDA_PREFIX}/epics/include/os/Linux
)

install(TARGETS epicstools
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
