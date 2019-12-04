add_library(digitizer SHARED
    src/Hsd.cc
    src/Stream.cc
)

target_link_libraries(digitizer
    utils
    xtcdata::xtc
)


target_include_directories(digitizer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES
    Hsd.hh
    HsdPython.hh
    Stream.hh
    DESTINATION include/psalg/digitizer
)

install(TARGETS digitizer
    EXPORT psalgTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
