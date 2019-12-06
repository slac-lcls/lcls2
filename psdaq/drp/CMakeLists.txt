find_package(PythonLibs REQUIRED)

add_library(drpbase
    DrpBase.cc
    FileWriter.cc
)

target_include_directories(drpbase PUBLIC
    ../psdaq/pgp/aes-stream-drivers/include
)

target_link_libraries(drpbase
    contributor
    exporter
    collection
)

add_executable(drp
    XpmDetector.cc
    AreaDetector.cc
    Digitizer.cc
    TimingSystem.cc
    TimeTool.cc
    PGPDetector.cc
    PGPDetectorApp.cc
    drp.cc
)

target_link_libraries(drp
    psalg::digitizer
    xtcdata::xtc
    pthread
    drpbase
    trigger
)

add_executable(drp_bld
    BldDetector.cc
)

target_link_libraries(drp_bld
    drpbase
    trigger
    epicstools
)

add_executable(drp_pva
    PvaDetector.cc
)

target_link_libraries(drp_pva
    drpbase
    trigger
    epicstools
)

add_executable(configdb_obj_2xtc
    configdb_obj_2xtc.cc
)

target_include_directories(configdb_obj_2xtc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${PYTHON_INCLUDE_DIRS}
    ${RAPIDJSON_INCLUDE_DIRS}
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
    ${RAPIDJSON_INCLUDE_DIRS}
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
    ../psdaq/pgp/aes-stream-drivers/include
)
target_link_libraries(pgpread
    xtcdata::xtc
)

add_executable(pgpread_timetool
    AxiBatcherParser.cc
    pgpread_timetool.cc
)
target_include_directories(pgpread_timetool PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ../psdaq/pgp/aes-stream-drivers/include
    ${PYTHON_INCLUDE_DIRS}
)
target_link_libraries(pgpread_timetool
    service
    xtcdata::xtc
    ${PYTHON_LIBRARIES}
)

add_executable(AxiBatcherParserTest
    AxiBatcherParserTest.cc
    AxiBatcherParser.cc
)
target_include_directories(AxiBatcherParserTest PUBLIC

    ../psdaq/pgp/aes-stream-drivers/include
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
     ../psdaq/pgp/aes-stream-drivers/include
 )
 target_link_libraries(drp_validate
     pthread
 )

install(TARGETS drp drp_bld drp_pva
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
