add_library(calib SHARED
    src/ArrayIO.cc
)

target_link_libraries(calib
    xtcdata::xtc
)


target_include_directories(calib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES
    ArrayIO.hh
    ArrayMetadata.hh
    DESTINATION include/psalg/calib
)

install(TARGETS calib
    EXPORT psalgTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
