add_library(detector SHARED
    src/DetectorTypes.cc
    src/Detector.cc
    src/DetectorStore.cc
    src/AreaDetectorTypes.cc
    src/AreaDetector.cc
    src/AreaDetectorEpix100a.cc
    src/AreaDetectorStore.cc
)

target_link_libraries(detector
    xtcdata::xtc
)

target_include_directories(detector PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES
    DetectorTypes.hh
    Detector.hh
    DetectorStore.hh
    AreaDetectorTypes.hh
    AreaDetector.hh
    AreaDetectorStore.hh
    AreaDetectorEpix100a.hh
    DataSourceSimulator.hh
    DESTINATION include/psalg/detector
)

install(TARGETS detector
    EXPORT psalgTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
