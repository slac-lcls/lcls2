
add_library(epicstools SHARED
    EpicsCA.cc
    PvServer.cc
)

target_include_directories(epicstools PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
#    $<INSTALL_INTERFACE:include>
    /reg/common/package/epicsca/3.14.12/include
    /reg/common/package/epicsca/3.14.12/include/os/Linux
)

#add_executable(h5writer
#    ../app/h5writer.cc
#)
#target_link_libraries(h5writer
#    pshdf5
#    xtc
#)
