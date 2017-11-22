
add_library(cphw SHARED
    Reg_rssi.cc
    AmcPLL.cc
    AmcTiming.cc
    AxiVersion.cc
    GthEyeScan.cc
    HsRepeater.cc
    RingBuffer.cc
    XBar.cc
)

add_library(cphwr SHARED
    Reg.cc
    AmcPLL.cc
    AmcTiming.cc
    AxiVersion.cc
    GthEyeScan.cc
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

#
#  amctiming
#
add_executable(amctiming amctiming.cc)

target_link_libraries(amctiming
    dl
    pthread
    cphw
    /reg/common/package/cpsw/framework-R3-4/x86_64-rhel7-opt/lib/libcpsw.so
#    /reg/common/package/boost/1.57.0/x86_64-rhel7-gcc48-opt/lib
    /reg/common/package/yaml-cpp/yaml-cpp-0.5.3/x86_64-rhel7-gcc48-opt/lib/libyaml-cpp.so
)

#
#  amctiming_udp
#
add_executable(amctiming_udp amctiming.cc)

target_link_libraries(amctiming_udp
    dl
    pthread
    cphwr
)

#
#  amc_eyescan
#
add_executable(amc_eyescan amc_eyescan.cc)

target_link_libraries(amc_eyescan
    dl
    pthread
    cphwr
#    /reg/common/package/cpsw/framework-R3-4/x86_64-rhel7-opt/lib/libcpsw.so
#    /reg/common/package/boost/1.57.0/x86_64-rhel7-gcc48-opt/lib
#    /reg/common/package/yaml-cpp/yaml-cpp-0.5.3/x86_64-rhel7-gcc48-opt/lib/libyaml-cpp.so
)

