
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
    $ENV{CONDA_PREFIX}/include
)

target_include_directories(cphwr PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)

find_library(YAML_CPP_LIBRARY yaml-cpp HINTS $ENV{CONDA_PREFIX}/lib)
find_library(CPSW_LIBRARY cpsw HINTS $ENV{CONDA_PREFIX}/lib)
#find_package(Boost 1.65 REQUIRED PATHS $ENV{CONDA_PREFIX})

#
#  amctiming
#
add_executable(amctiming amctiming.cc)

target_link_libraries(amctiming
    cphw
    ${CPSW_LIBRARY}
    ${YAML_CPP_LIBRARY}
    dl
    pthread
    rt
#    /reg/common/package/cpsw/framework-R3-4/x86_64-rhel7-opt/lib/libcpsw.so
#    /reg/common/package/boost/1.57.0/x86_64-rhel7-gcc48-opt/lib
#    /reg/common/package/yaml-cpp/yaml-cpp-0.5.3/x86_64-rhel7-gcc48-opt/lib/libyaml-cpp.so
)

#
#  amctiming_udp
#
add_executable(amctiming_udp amctiming.cc)

target_link_libraries(amctiming_udp
    cphwr
    dl
    pthread
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

