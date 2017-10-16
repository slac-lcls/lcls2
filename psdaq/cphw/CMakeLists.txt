
add_library(cphw SHARED
    Reg_rssi.cc
    AmcTiming.cc
    AxiVersion.cc
    HsRepeater.cc
    RingBuffer.cc
    XBar.cc
)

add_library(cphwr SHARED
    Reg.cc
    AmcTiming.cc
    AxiVersion.cc
    HsRepeater.cc
    RingBuffer.cc
    XBar.cc
)

target_include_directories(cphw PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    /reg/common/package/cpsw/framework-R3-4/x86_64-rhel7-opt/include
    /reg/common/package/boost/1.57.0/x86_64-rhel7-gcc48-opt/include
    /reg/common/package/yaml-cpp/yaml-cpp-0.5.3/x86_64-rhel7-gcc48-opt/include
)

target_include_directories(cphwr PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    /reg/common/package/cpsw/framework-R3-4/x86_64-rhel7-opt/include
    /reg/common/package/boost/1.57.0/x86_64-rhel7-gcc48-opt/include
    /reg/common/package/yaml-cpp/yaml-cpp-0.5.3/x86_64-rhel7-gcc48-opt/include
)

#add_executable(h5writer
#    ../app/h5writer.cc
#)
#target_link_libraries(h5writer
#    pshdf5
#    xtc
#)
