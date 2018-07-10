add_executable(xtcreader
    xtcreader.cc
)
target_link_libraries(xtcreader
    xtc
)

add_executable(xtcwriter
    xtcwriter.cc
)
target_link_libraries(xtcwriter
    xtc
)

add_executable(xtcsave
    xtcsave.cc
)
target_link_libraries(xtcsave
    xtc
)

add_executable(smdwriter
    smdwriter.cc
)

target_link_libraries(smdwriter
    xtc
)

install(TARGETS xtcwriter smdwriter xtcreader
    EXPORT xtcdataTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
