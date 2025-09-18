add_library(calib SHARED
    src/ArrayIO.cc
    src/CalibParsTypes.cc
    src/AreaDetectorTypes.cc
    src/CalibPars.cc
    src/CalibParsDB.cc
    src/CalibParsDBTypes.cc
    src/CalibParsDBWeb.cc
    src/CalibParsDBStore.cc
    src/CalibParsEpix100a.cc
    src/CalibParsStore.cc
    src/Query.cc
    src/MDBWebUtils.cc
)

target_include_directories(calib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${RapidJSON_INCLUDE_DIRS}
)

target_link_libraries(calib
    utils
    geometry
    xtcdata::xtc
    CURL::libcurl
)

install(FILES
    Types.hh
    AreaDetectorTypes.hh
    NDArray.hh
    ArrayIO.hh
    CalibParsTypes.hh
    CalibParsDBTypes.hh
    CalibPars.hh
    CalibParsDB.hh
    CalibParsDBWeb.hh
    CalibParsDBStore.hh
    CalibParsEpix100a.hh
    CalibParsStore.hh
    Query.hh
    MDBWebUtils.hh
    DESTINATION include/psalg/calib
)

## Test CalibPars
#add_executable(test_CalibPars
#    ../tests/test_CalibPars.cc
#)
#target_link_libraries(test_CalibPars
#    psalg
#    calib
#    xtcdata::xtc
#)

install(TARGETS calib
    EXPORT psalgTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
