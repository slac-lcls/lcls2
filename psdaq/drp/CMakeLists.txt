add_library(drp SHARED
    #Digitizer.cc
    AreaDetector.cc
    Worker.cc
    Collector.cc
    PGPReader.cc
    drp.cc
)

target_link_libraries(drp
    xtcdata
    pgpdriver
    #pshdf5
    eb
    $ENV{CONDA_PREFIX}/lib/libzmq.so
    $ENV{CONDA_PREFIX}/lib/libstdc++.so.6
)

target_include_directories(drp PUBLIC
    ../psdaq/pgp/user-driver
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

add_executable(main
    main2.cc
)

target_link_libraries(main
    drp
    pthread
)
