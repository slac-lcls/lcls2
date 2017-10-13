
add_library(pshdf5 SHARED
    src/Hdf5Writer.cc
)

target_include_directories(pshdf5 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(pshdf5
    ${HDF5_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
)

add_executable(h5writer
    ../app/h5writer.cc
)
target_link_libraries(h5writer
    pshdf5
    xtc
)
