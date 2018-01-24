add_library(drp SHARED
    Digitizer.cc
    AreaDetector.cc
    EventBuilder.cc
)

target_link_libraries(drp
    xtc
    pshdf5
    eb
)

add_executable(main
    main.cc
)

target_include_directories(main PUBLIC
    ../pgp/pgpcardG3/include/
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(main
    drp
    pthread
)
