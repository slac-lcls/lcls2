
find_package(OpenMP REQUIRED)

add_library(geometry SHARED
    src/SegGeometry.cc
    src/SegGeometryCspad2x1V1.cc
    src/SegGeometryEpix100V1.cc
    src/SegGeometryEpix10kaV1.cc
    src/SegGeometryMatrixV1.cc
    src/GeometryObject.cc
    src/GeometryAccess.cc
)

target_compile_options(geometry PRIVATE ${OpenMP_CXX_FLAGS})

target_include_directories(geometry PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
)


target_link_libraries(geometry ${OpenMP_CXX_FLAGS}
    utils
    xtc
    CURL::libcurl
)

install(FILES
    GeometryTypes.hh
    SegGeometry.hh
    SegGeometryCspad2x1V1.hh
    SegGeometryEpix100V1.hh
    SegGeometryEpix10kaV1.hh
    SegGeometryMatrixV1.hh
    SegGeometryStore.hh
    UtilsCSPAD.hh
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
#    xtc
#)

install(TARGETS geometry
		#test_CalibPars # this installs under psalg/build/psalg/calib/
    EXPORT psalgTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
