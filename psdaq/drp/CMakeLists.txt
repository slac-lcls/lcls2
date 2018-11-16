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
    ../../../aes-stream-drivers/include
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

#add_executable(pgpread2
#    pgpread.cc
#)

#target_include_directories(pgpread2 PUBLIC
#    ../../../aes-stream-drivers/include
#)

#target_link_libraries(pgpread2
#    xtcdata::xtc
#)
