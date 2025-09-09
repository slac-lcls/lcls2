find_package(PythonLibs REQUIRED)
find_package(nlohmann_json REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(drpbase
    PythonConfigScanner.cc
    BEBDetector.cc
    TTFex.cc
    XpmDetector.cc
    DrpBase.cc
    FileWriter.cc
    Si570.cc
)

target_include_directories(drpbase PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(drpbase
    contributor
    exporter
    collection
    xtcdata::xtc
    epicstools
    Threads::Threads
    nlohmann_json::nlohmann_json
)

# Specify the directory where the shared libraries are located
set(SC_TDC_LIB_DIR ${CMAKE_SOURCE_DIR}/libs/scTDC)

# Create a library for PipeCallbackHandler
add_library(pipecallbackhandler_lib
    PipeCallbackHandler.cc
)

# Include directories for the library
target_include_directories(pipecallbackhandler_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/scTDC>
)

# Link the shared libraries to pipecallbackhandler_lib
target_link_directories(pipecallbackhandler_lib PUBLIC ${SC_TDC_LIB_DIR})

# Set RPATH for runtime library loading
set_target_properties(pipecallbackhandler_lib PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH ${SC_TDC_LIB_DIR}
)

# Install the shared libraries to the target location
install(FILES ${SC_TDC_LIB_DIR}/libscTDC.so
        DESTINATION lib)

# Link dependencies needed by PipeCallbackHandler
target_link_libraries(pipecallbackhandler_lib
    drpbase
    Threads::Threads
    xtcdata::xtc
    nlohmann_json::nlohmann_json
    epicstools
    ${SC_TDC_LIB_DIR}/libscTDC.so
)

# Create a library for BufferedDataCallbackHandler
add_library(buffereddatacallbackhandler_lib
    BufferedDataCallbackHandler.cc
)

target_include_directories(buffereddatacallbackhandler_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/scTDC>
)

target_link_directories(buffereddatacallbackhandler_lib PUBLIC ${SC_TDC_LIB_DIR})

set_target_properties(buffereddatacallbackhandler_lib PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH ${SC_TDC_LIB_DIR}
)

target_link_libraries(buffereddatacallbackhandler_lib
    drpbase
    Threads::Threads
    xtcdata::xtc
    nlohmann_json::nlohmann_json
    epicstools
    ${SC_TDC_LIB_DIR}/libscTDC.so
)

add_executable(drp
    AreaDetector.cc
    Digitizer.cc
    EpixHR2x2.cc
    EpixHRemu.cc
    EpixM320.cc
    EpixUHR.cc
    Epix100.cc
    EpixQuad.cc
    Jungfrau.cc
    JungfrauDetectorId.cc
    JungfrauEmulator.cc
    TimingDef.cc
    TimingBEB.cc
    TimingSystem.cc
    TimeTool.cc
    Wave8.cc
    HREncoder.cc
    Opal.cc
#    OpalTT.cc
    OpalTTFex.cc
#    OpalTTSim.cc
    Piranha4.cc
    Piranha4TTFex.cc
    TebReceiver.cc
    PGPDetector.cc
    PGPDetectorApp.cc
    drp.cc
)

target_link_libraries(drp
    psalg::digitizer
    psalg::detector
    xtcdata::xtc
    drpbase
    trigger
    mmhw
    nlohmann_json::nlohmann_json
    slsDetectorShared
)

add_executable(drp_bld
    TebReceiver.cc
#    BldDetectorSlow.cc
    BldDetector.cc
    BldNames.cc
)

target_include_directories(drp_bld PUBLIC
    ${Readline_INCLUDE_DIR}
)

target_link_libraries(drp_bld
    pipecallbackhandler_lib
    drpbase
    trigger
    epicstools
    mmhw
    ${Readline_LIBRARY}
)

add_executable(drp_pva
    TebReceiver.cc
    PvaDetector.cc
)

target_include_directories(drp_pva PUBLIC
    ${Readline_INCLUDE_DIR}
)

target_link_libraries(drp_pva
    drpbase
    trigger
    mmhw
    epicstools
    ${Readline_LIBRARY}
)

add_executable(drp_udpencoder
    TebReceiver.cc
    UdpEncoder.cc
)

target_include_directories(drp_udpencoder PUBLIC
    ${Readline_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(drp_udpencoder
    drpbase
    trigger
    mmhw
    ${Readline_LIBRARY}
)

add_executable(tpr_trigger
    TprTrigger.cc
)

target_link_libraries(tpr_trigger
    collection
    app
    tpr
    dl
    ${Readline_LIBRARY}
    Threads::Threads
    rt
)

add_executable(drp_kmicro_test
    test_kmicro.cc
)

target_link_libraries(drp_kmicro_test
    pipecallbackhandler_lib
    drpbase
    trigger
    epicstools
    mmhw
    ${Readline_LIBRARY}
)

add_executable(configdb_obj_2xtc
    configdb_obj_2xtc.cc
)

target_include_directories(configdb_obj_2xtc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${PYTHON_INCLUDE_DIRS}
    ${RapidJSON_INCLUDE_DIRS}
)

target_link_libraries(configdb_obj_2xtc
    service
    xtcdata::xtc
    ${PYTHON_LIBRARIES}
)

add_executable(pgpread
    pgpread.cc
)
target_include_directories(pgpread PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)
target_link_libraries(pgpread
    mmhw
    psalg::utils
    xtcdata::xtc
    Threads::Threads
)

add_executable(pgpread_timetool
    AxiBatcherParser.cc
    pgpread_timetool.cc
)
target_include_directories(pgpread_timetool PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${PYTHON_INCLUDE_DIRS}
)
target_link_libraries(pgpread_timetool
    service
    xtcdata::xtc
    ${PYTHON_LIBRARIES}
    Threads::Threads
    rt
)

add_executable(pgpread_jungfrau
    pgpread_jungfrau.cc
)
target_include_directories(pgpread_jungfrau PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)
target_link_libraries(pgpread_jungfrau
    mmhw
    psalg::utils
    xtcdata::xtc
    Threads::Threads
)

add_executable(pgpread_epixM320mon
    pgpread_epixM320mon.cc
)
target_include_directories(pgpread_epixM320mon PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)
target_link_libraries(pgpread_epixM320mon
    Threads::Threads
)

add_executable(AxiBatcherParserTest
    AxiBatcherParserTest.cc
    AxiBatcherParser.cc
)
target_include_directories(AxiBatcherParserTest PUBLIC
    ../psdaq/drp
    ${PYTHON_INCLUDE_DIRS}
)
target_link_libraries(AxiBatcherParserTest
    xtcdata::xtc
    drpbase
    ${PYTHON_LIBRARIES}
)

add_executable(drp_validate
    validate.cc
)
target_include_directories(drp_validate PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)
target_link_libraries(drp_validate
    Threads::Threads
)

add_executable(opaltt_test
    opaltt_test.cc
)

target_include_directories(opaltt_test PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(opaltt_test
    xtcdata::xtc
    psalg::calib
)

add_executable(fileWriteTest
    fileWriteTest.cc
)

target_link_libraries(fileWriteTest
    drpbase
    Threads::Threads
)

add_executable(drp_groupsync
    groupsync.cc
)
target_include_directories(drp_groupsync PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)
target_link_libraries(drp_groupsync
    Threads::Threads
)

add_executable(ePixMDescTest
    ePixMDescTest.cc
)

add_executable(jungfrauTest
    jungfrauTest.cc
    JungfrauDetectorId.cc
)

target_include_directories(jungfrauTest PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(jungfrauTest
    slsDetectorShared
)

add_executable(jungfrauSim
    jungfrauSim.cc
)

target_include_directories(jungfrauSim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)
target_link_libraries(jungfrauSim
    xtcdata::xtc
)


install(TARGETS drp
  drp_bld
  drp_pva
  drp_udpencoder
  tpr_trigger
  drp_kmicro_test
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
