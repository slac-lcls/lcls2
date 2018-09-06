
add_library(cphw SHARED
    Reg_rssi.cc
    AmcPLL.cc
    AmcTiming.cc
    AxiVersion.cc
    GthEyeScan.cc
    HsRepeater.cc
    RingBuffer.cc
    XBar.cc
    BldControl.cc
    Xvc.cc
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
    BldControl.cc
    Xvc.cc
)

target_include_directories(cphw PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)

target_include_directories(cphwr PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)

find_library(YAML_CPP_LIBRARY yaml-cpp PATH $ENV{CONDA_PREFIX}/lib)
find_library(CPSW_LIBRARY cpsw PATH $ENV{CONDA_PREFIX}/lib)
#find_package(Boost 1.65 REQUIRED PATHS $ENV{CONDA_PREFIX})

#
#  amc_timing
#
add_executable(amc_timing amc_timing.cc)

target_link_libraries(amc_timing
    cphw
    ${CPSW_LIBRARY}
    ${YAML_CPP_LIBRARY}
    dl
    pthread
    rt
)

#
#  amctiming_udp
#
add_executable(amc_timing_udp amc_timing.cc)

target_link_libraries(amc_timing_udp
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
)

#
#  hsrCfg
#
add_executable(hsrCfg hsrCfg.cc)

target_link_libraries(hsrCfg
    cphwr
    ${CPSW_LIBRARY}
    ${YAML_CPP_LIBRARY}
    dl
    pthread
    rt
)

#
#  regtest
#
add_executable(regtest regtest.cc)

target_link_libraries(regtest
    dl
    pthread
    cphwr
)


install(TARGETS cphw
                amc_timing
                amc_timing_udp
                amc_eyescan
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
