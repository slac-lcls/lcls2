
add_library(psdaq SHARED
    hdf5/src/Hdf5Writer.cc
)

target_include_directories(psdaq PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(psdaq
    ${HDF5_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
)

add_executable(h5writer
    app/h5writer.cc
)
target_link_libraries(h5writer
    psdaq
    xtc
)
