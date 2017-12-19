cmake_minimum_required(VERSION 3.2 FATAL_ERROR)



add_library(drp SHARED
    Digitizer.cc
    AreaDetector.cc
)

target_link_libraries(drp
    xtc
    pshdf5
    eb
)

add_executable(main
    main.cc
)

#target_include_directories(utils PUBLIC
#    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
#    $<INSTALL_INTERFACE:include>
#)

target_link_libraries(main
    drp
    pthread
    xtc
    pshdf5
    eb
)
