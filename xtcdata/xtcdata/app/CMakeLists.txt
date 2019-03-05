add_executable(xtcreader
    xtcreader.cc
)
target_link_libraries(xtcreader
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
target_include_directories(xtcwriter PUBLIC
    $ENV{CONDA_PREFIX}/include
)

add_executable(amiwriter
    amiwriter.cc
)
target_link_libraries(amiwriter
    xtc
)
target_include_directories(amiwriter PUBLIC
    $ENV{CONDA_PREFIX}/include
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

add_executable(json2xtc
    json2xtc.cc
)

target_link_libraries(json2xtc
    j2x xtc
)

install(TARGETS xtcwriter smdwriter xtcreader json2xtc amiwriter
    EXPORT xtcdataTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
