add_library(utils SHARED
    Digitizer.cc
    AreaDetector.cc
    Worker.cc
    Collector.cc
    PGPReader.cc
    utils.cc
)

find_package(PythonLibs REQUIRED)

target_link_libraries(utils
    xtcdata::xtc
    pgpdriver
    eb
    zmq
    ${PYTHON_LIBRARIES}
)

target_include_directories(utils PUBLIC
    ../psdaq/pgp/user-driver
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${PYTHON_INCLUDE_DIRS}
    $ENV{CONDA_PREFIX}/include
)

add_executable(drp
    drp.cc
)

target_link_libraries(drp
    utils
    pthread
    collection
)
