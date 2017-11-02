target_include_directories(xtc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

add_executable(h5writer
    h5writer.cc
)
target_link_libraries(h5writer
    pshdf5
    xtc
)
