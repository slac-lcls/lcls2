add_library(calib SHARED
    src/ArrayIO.cc
    src/AreaDetectorTypes.cc
    src/CalibPars.cc
    src/CalibParsDB.cc
    src/CalibParsDBWeb.cc
    src/CalibParsDBStore.cc
    src/CalibParsEpix100a.cc
    src/CalibParsStore.cc
    src/MDBWebUtils.cc
)

target_include_directories(calib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:$ENV{CONDA_PREFIX}/include> # for expernal libs
)

find_library(CURL_LIBRARY curl PATH $ENV{CONDA_PREFIX}/lib)

target_link_libraries(calib
    xtcdata::xtc
    ${CURL_LIBRARY}
)

install(FILES
    Types.hh
    AreaDetectorTypes.hh
    NDArray.hh
    ArrayIO.hh
    CalibParsTypes.hh
    CalibPars.hh
    CalibParsDB.hh
    CalibParsDBWeb.hh
    CalibParsDBStore.hh
    CalibParsEpix100a.hh
    CalibParsStore.hh
    Query.hh
    Response.hh
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
		#test_CalibPars # this installs under psalg/build/psalg/calib/
    EXPORT psalgTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
