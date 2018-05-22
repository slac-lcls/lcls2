add_library(utils SHARED
    Digitizer.cc
    AreaDetector.cc
    Worker.cc
    Collector.cc
    PGPReader.cc
    utils.cc
)

target_link_libraries(utils
    xtcdata
    pgpdriver
    #pshdf5
    eb
    $ENV{CONDA_PREFIX}/lib/libzmq.so
    $ENV{CONDA_PREFIX}/lib/libstdc++.so.6
)

target_include_directories(utils PUBLIC
    ../psdaq/pgp/user-driver
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

add_executable(drp
    drp.cc
)

target_link_libraries(drp
    utils
    pthread
)
