find_package(PythonLibs REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(drpbase
    PythonConfigScanner.cc
    BEBDetector.cc
    XpmDetector.cc
    DrpBase.cc
    FileWriter.cc
    Si570.cc
)

target_include_directories(drpbase PUBLIC
    ${AES_STREAM_DRIVER_HEADER_PATH}
)

target_link_libraries(drpbase
    contributor
    exporter
    collection
    xtcdata::xtc
    epicstools
    Threads::Threads
)

add_executable(drp
    AreaDetector.cc
    Digitizer.cc
    EpixHR2x2.cc
    EpixHRemu.cc
    EpixM320.cc
    Epix100.cc
    EpixQuad.cc
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
)

add_executable(drp_bld
#    BldDetectorSlow.cc
    BldDetector.cc
    BldNames.cc
)

target_include_directories(drp_bld PUBLIC
    ${Readline_INCLUDE_DIR}
)

target_link_libraries(drp_bld
    drpbase
    trigger
    epicstools
    ${Readline_LIBRARY}
)

add_executable(drp_pva
    PvaDetector.cc
)

target_include_directories(drp_pva PUBLIC
    ${Readline_INCLUDE_DIR}
)

target_link_libraries(drp_pva
    drpbase
    trigger
    epicstools
    ${Readline_LIBRARY}
)

add_executable(drp_udpencoder
    UdpEncoder.cc
)

target_include_directories(drp_udpencoder PUBLIC
    ${Readline_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(drp_udpencoder
    drpbase
    trigger
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
    readline
    Threads::Threads
    rt
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

add_executable(test_teb_config
    test_teb_config.cc
)

target_include_directories(test_teb_config PUBLIC
    ${RapidJSON_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
)

target_link_libraries(test_teb_config
    ${PYTHON_LIBRARIES}
)

add_executable(pgpread
    pgpread.cc
)
target_include_directories(pgpread PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    ${AES_STREAM_DRIVER_HEADER_PATH}
)
target_link_libraries(pgpread
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
    ${AES_STREAM_DRIVER_HEADER_PATH}
    ${PYTHON_INCLUDE_DIRS}
)
target_link_libraries(pgpread_timetool
    service
    xtcdata::xtc
    ${PYTHON_LIBRARIES}
    Threads::Threads
    rt
)

add_executable(AxiBatcherParserTest
    AxiBatcherParserTest.cc
    AxiBatcherParser.cc
)
target_include_directories(AxiBatcherParserTest PUBLIC

    ${AES_STREAM_DRIVER_HEADER_PATH}
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
    ${AES_STREAM_DRIVER_HEADER_PATH}
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
    ${AES_STREAM_DRIVER_HEADER_PATH}
)
target_link_libraries(drp_groupsync
    Threads::Threads
)

if(CMAKE_CUDA_COMPILER)
    add_executable(test_gpu_add
        test_gpu_add.cu
    )
    set_target_properties(test_gpu_add PROPERTIES CUDA_ARCHITECTURES "86")
endif()


install(TARGETS drp
  drp_bld
  drp_pva
  drp_udpencoder
  tpr_trigger
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
