find_package(HDF5 REQUIRED HL)

add_library(pshdf5 SHARED
    src/Hdf5Writer.cc
)

target_include_directories(pshdf5 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(pshdf5
    xtcdata
    ${HDF5_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
    $ENV{CONDA_PREFIX}/lib/libstdc++.so.6
)

install(TARGETS pshdf5
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
