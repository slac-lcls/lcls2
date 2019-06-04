add_library(geometry SHARED
    src/SegGeometry.cc
    src/SegGeometryCspad2x1V1.cc
    src/SegGeometryEpix100V1.cc
    src/SegGeometryEpix10kaV1.cc
    src/SegGeometryMatrixV1.cc
    src/GeometryObject.cc
    src/GeometryAccess.cc
)

target_include_directories(geometry PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:$ENV{CONDA_PREFIX}/include> # for expernal libs
)

find_library(CURL_LIBRARY curl PATH $ENV{CONDA_PREFIX}/lib)

target_link_libraries(geometry
    xtcdata::xtc
    calib
    ${CURL_LIBRARY}
)

install(FILES
    SegGeometryCspad2x1V1.hh
    SegGeometryEpix100V1.hh
    SegGeometryEpix10kaV1.hh
    SegGeometry.hh
    SegGeometryMatrixV1.hh
    SegGeometryStore.hh
    GlobalMethods.hh
    GeometryObject.hh
    GeometryAccess.hh
    DESTINATION include/psalg/geometry
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

install(TARGETS geometry
		#test_CalibPars # this installs under psalg/build/psalg/calib/
    EXPORT psalgTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
