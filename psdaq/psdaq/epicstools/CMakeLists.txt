
add_library(epicstools SHARED
    EpicsPVA.cc
    PvServer.cc
)

target_include_directories(epicstools PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/epics/include
    $ENV{CONDA_PREFIX}/epics/include/os/Linux
    $ENV{CONDA_PREFIX}/epics/include/compiler/gcc
)

target_link_libraries(epicstools
    ${EPICS_PVACCESS_LIB}
    ${EPICS_PVDATA_LIB}
)

install(TARGETS epicstools
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
