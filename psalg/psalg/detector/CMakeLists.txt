
#   src/AreaDetectorTypes.cc - moved to calib for correct dependences
add_library(detector SHARED
    src/DetectorTypes.cc
    src/Detector.cc
    src/DetectorStore.cc
    src/UtilsConfig.cc
    src/AreaDetector.cc
    src/AreaDetectorJungfrau.cc
    src/AreaDetectorPnccd.cc
    src/AreaDetectorOpal.cc
    src/AreaDetectorEpix100a.cc
    #src/AreaDetectorCspad.cc
    src/AreaDetectorStore.cc
)

target_link_libraries(detector
    xtc
    calib
)

target_include_directories(detector PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
)

#   AreaDetectorTypes.hh - moved to calib for correct dependences
install(FILES
    DetectorTypes.hh
    Detector.hh
    DetectorStore.hh
    UtilsConfig.hh
    AreaDetector.hh
    AreaDetectorStore.hh
    AreaDetectorJungfrau.hh
    AreaDetectorPnccd.hh
    AreaDetectorOpal.hh
    AreaDetectorEpix100a.hh
    #AreaDetectorCspad.hh
    DataSourceSimulator.hh
    DESTINATION include/psalg/detector
)

install(TARGETS detector
    EXPORT psalgTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
