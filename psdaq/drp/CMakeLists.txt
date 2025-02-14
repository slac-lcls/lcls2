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
    mmhw
    nlohmann_json::nlohmann_json
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
    mmhw
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
    mmhw
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

if(CMAKE_CUDA_COMPILER)
    #message(CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")
    #message(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES="${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    #message(CMAKE_CXX_COMPILER='${CMAKE_CXX_COMPILER}')
    #message(NVCC_PREPEND_FLAGS='$ENV{NVCC_PREPEND_FLAGS}')

    add_executable(drp_gpu
        GpuDetector.cc
        GpuDetectorApp.cc
    )
    # @todo: No need for this anymore, I think
    #target_include_directories(drp_gpu PUBLIC
    #    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    #)
    target_link_libraries(drp_gpu
        xtcdata::xtc
        drpbase
        trigger
        mmhw
        nlohmann_json::nlohmann_json
    )

    add_library(AreaDetector_gpu SHARED
        AreaDetectorGpu.cc
        GpuWorker_impl.cu
        AreaDetector.cc
    )
    target_include_directories(AreaDetector_gpu PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        #$<INSTALL_INTERFACE:include>  # Not working?
        ${CMAKE_INSTALL_PREFIX}/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(AreaDetector_gpu
        xtcdata::xtc
        drpbase
        trigger  # @todo: This needed?
        mmhw
        nlohmann_json::nlohmann_json  # @todo: This needed?
        CUDA::cudart
        CUDA::cuda_driver
    )
    set_target_properties(AreaDetector_gpu PROPERTIES CUDA_ARCHITECTURES "86")

    add_executable(pgpreadgpu
        pgpreadgpu.cc
        GpuAsyncLib.cc
    )
    target_include_directories(pgpreadgpu PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(pgpreadgpu
        CUDA::cudart
        CUDA::cuda_driver
        Threads::Threads
    )
endif()

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

install(TARGETS drp
  drp_bld
  drp_pva
  drp_udpencoder
  tpr_trigger
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(CMAKE_CUDA_COMPILER)
  install(TARGETS
    drp_gpu
    AreaDetector_gpu
  )
endif()
