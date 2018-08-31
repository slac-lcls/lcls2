add_library(app SHARED
    XtcMonitorServer.cc
    XtcMonitorClient.cc
    XtcMonitorMsg.cc
    TransitionCache.cc
)

target_include_directories(app PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

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

add_executable(xtcMonClient
    xtcMonClient.cc
)
target_link_libraries(xtcMonClient
    xtc
    app
    rt
)

add_executable(xtcMonWriter
    xtcMonWriter.cc
)
target_link_libraries(xtcMonWriter
    xtc
    app
    rt
)

install(FILES
    TransitionCache.hh
    XtcMonitorMsg.hh
    XtcMonitorClient.hh
    XtcMonitorServer.hh
    DESTINATION include/xtcdata/app
)

install(TARGETS
    app
    xtcwriter
    smdwriter
    xtcreader
    xtcMonClient
    xtcMonWriter
    EXPORT xtcdataTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
