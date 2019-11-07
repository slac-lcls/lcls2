add_executable(xtcreader
    xtcreader.cc
)
target_link_libraries(xtcreader
    xtc
)

add_executable(randomaccess
    randomaccess.cc
)
target_link_libraries(randomaccess
    xtc
)

add_executable(jungfrau
    jungfrau.cc
)
target_link_libraries(jungfrau
    xtc
)

add_executable(xtcwriter
    xtcwriter.cc
)
target_link_libraries(xtcwriter
    xtc
)

add_executable(amiwriter
    amiwriter.cc
)
target_link_libraries(amiwriter
    xtc
)

add_executable(xtcsave
    xtcsave.cc
)
target_link_libraries(xtcsave
    xtc
)

add_executable(hsdduplicate
    hsdduplicate.cc
)
target_link_libraries(hsdduplicate
    xtc
)

add_executable(smdwriter
    smdwriter.cc
)

target_link_libraries(smdwriter
    xtc
)

install(TARGETS xtcwriter smdwriter xtcreader amiwriter
    EXPORT xtcdataTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
