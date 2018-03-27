add_library(drp_olddriver SHARED
    Digitizer.cc
    #AreaDetector.cc
    #EventBuilder.cc
)

target_link_libraries(drp_olddriver
    xtcdata
    pshdf5
    eb
)

add_executable(main_olddriver
    main.cc
)

target_include_directories(main PUBLIC
    ../pgp/pgpcardG3/include/
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(main_olddriver
    drp_olddriver
    pthread
)
