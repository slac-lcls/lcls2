add_library(drputils SHARED
    Digitizer.cc
    AreaDetector.cc
    Worker.cc
    Collector.cc
    PGPReader.cc
    utils.cc
)

find_package(PythonLibs REQUIRED)

target_link_libraries(drputils
    xtcdata::xtc
    pgpdriver
    eb
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
    drp.cc
)

target_link_libraries(drp
    drputils
    pthread
    collection
)

install(TARGETS drputils drp
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# TEST
add_executable(drp2 DrpApp.cc DrpApp.hh main.cc)
target_include_directories(drp2 PUBLIC
    $ENV{CONDA_PREFIX}/include)

target_link_libraries(drp2
    pthread
    drputils
    collection
    eb
    zmq)

add_executable(pgpread
    pgpread.cc
)

target_include_directories(pgpread PUBLIC
    ../psdaq/pgp/aes-stream-drivers/include
)

target_link_libraries(pgpread
    xtcdata::xtc
)
