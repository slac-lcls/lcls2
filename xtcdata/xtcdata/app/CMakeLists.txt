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
