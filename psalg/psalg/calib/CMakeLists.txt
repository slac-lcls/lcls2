add_library(calib SHARED
    src/ArrayIO.cc
    src/CalibPars.cc
    src/CalibParsStore.cc
)

target_link_libraries(calib
    xtcdata::xtc
)

target_include_directories(calib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES
    Types.hh
    NDArray.hh
    ArrayIO.hh
    CalibPars.hh
    CalibParsTypes.hh
    CalibParsStore.hh
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
