
add_library(cphw SHARED
    Reg_rssi.cc
    AmcPLL.cc
    AmcTiming.cc
    AxiVersion.cc
    GthEyeScan.cc
    GthRxAlign.cc
    HsRepeater.cc
    Logging.cc
    RingBuffer.cc
    TimingRx.cc
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
    GthRxAlign.cc
    HsRepeater.cc
    Logging.cc
    RingBuffer.cc
    TimingRx.cc
    XBar.cc
    BldControl.cc
    Xvc.cc
)

target_include_directories(cphw PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_include_directories(cphwr PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

#find_package(Boost 1.65 REQUIRED PATHS $ENV{CONDA_PREFIX})

#
#  amc_timing
#
#add_executable(amc_timing amc_timing.cc)

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
    -pthread
)


#
#  regtest
#
add_executable(xvc_srv xvc_srv.cc)

target_link_libraries(xvc_srv
    dl
    pthread
    cphwr
    -pthread
)


install(TARGETS cphw
                cphwr
#                amc_timing
                amc_timing_udp
                amc_eyescan
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
