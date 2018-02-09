
add_executable(h5writer
    h5writer.cc
)
target_link_libraries(h5writer
    pshdf5
    xtc
)
