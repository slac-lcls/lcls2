add_library(peaks SHARED
    src/LocalExtrema.cc
    src/PeakFinderAlgos.cc
)

target_link_libraries(peaks
    xtcdata::xtc
)


target_include_directories(peaks PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES
    LocalExtrema.hh
    PeakFinderAlgos.hh
    Types.hh
    DESTINATION include/psalg/peaks
)

install(TARGETS peaks
    EXPORT psalgTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
