add_executable(xtcreader
    xtcreader.cc
)
target_link_libraries(xtcreader
    xtc
)

add_executable(xtcwriter
    xtcwriter.cc
)

find_package(PythonLibs 3.6 REQUIRED)
target_link_libraries(xtcwriter
    xtc
    ${PYTHON_LIBRARIES}
)
target_include_directories(xtcwriter PUBLIC
    $ENV{CONDA_PREFIX}/include
    ${PYTHON_INCLUDE_DIRS}
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
