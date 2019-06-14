find_package(PythonLibs REQUIRED)

add_executable(drp
    PGPReader.cc
    DrpApp.cc
    XpmDetector.cc
    AreaDetector.cc
    Digitizer.cc
    TimingSystem.cc
    TimeTool.cc
    FileWriter.cc
    drp.cc
)

target_include_directories(drp PUBLIC
    ../psdaq/pgp/aes-stream-drivers/include
    $ENV{CONDA_PREFIX}/include
)

target_link_libraries(drp
    xtcdata::xtc
    xtcdata::j2x
    pthread
    zmq
    collection
    contributor
    exporter
)

add_executable(test_hsd_config
    test_hsd_config.cc
)

target_include_directories(test_hsd_config PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    ${PYTHON_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/include)

target_link_libraries(test_hsd_config
    xtcdata::j2x
    xtcdata::xtc
    ${PYTHON_LIBRARIES}
)

add_executable(test_teb_config
    test_teb_config.cc
)

target_include_directories(test_teb_config PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    ${PYTHON_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/include)

target_link_libraries(test_teb_config
    xtcdata::j2x
    xtcdata::xtc
    ${PYTHON_LIBRARIES}
)

add_executable(pgpread
    pgpread.cc
)
target_include_directories(pgpread PUBLIC
    ../psdaq/pgp/aes-stream-drivers/include
)
target_link_libraries(pgpread
    xtcdata::xtc
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

install(TARGETS drp
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
