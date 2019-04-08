add_library(drputils SHARED
    Digitizer.cc
    AreaDetector.cc
    Worker.cc
    PGPReader.cc
    utils.cc
)

find_package(PythonLibs REQUIRED)

target_link_libraries(drputils
    xtcdata::xtc
    pgpdriver
    ctrb
    zmq
    ${PYTHON_LIBRARIES}
)

target_include_directories(drputils PUBLIC
    ../psdaq/pgp/aes-stream-drivers/include
    ../psdaq/service
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${PYTHON_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/include
)

add_executable(drp
    DrpApp.cc
    DrpApp.hh
    drp.cc
)

target_include_directories(drp PUBLIC
    $ENV{CONDA_PREFIX}/include)

target_link_libraries(drp
    pthread
    drputils
    collection
    ctrb
    zmq
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

add_executable(pgpread
    pgpread.cc
)

target_include_directories(pgpread PUBLIC
    ../psdaq/pgp/aes-stream-drivers/include
)

target_link_libraries(pgpread
    xtcdata::xtc
)

add_executable(tdetsim2
    tdetsim.cc
)

target_include_directories(tdetsim2 PUBLIC
    ../psdaq/pgp/aes-stream-drivers/include
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

install(TARGETS drputils drp
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
