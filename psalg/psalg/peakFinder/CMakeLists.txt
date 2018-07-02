add_library(peakFinder SHARED
    src/LocalExtrema.cc
    src/PeakFinderAlgos.cc
)

target_link_libraries(peakFinder
    xtcdata::xtc
)


target_include_directories(peakFinder PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES
    LocalExtrema.hh
    PeakFinderAlgos.hh
    Types.hh
    DESTINATION include/psalg/peakFinder
)

install(TARGETS peakFinder
    EXPORT psalgTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
