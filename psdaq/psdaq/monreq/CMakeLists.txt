add_library(monreq SHARED
    XtcMonitorServer.cc
    XtcMonitorClient.cc
    XtcMonitorMsg.cc
    TransitionCache.cc
)

target_include_directories(monreq PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(monreq
    xtcdata::xtc
)

add_executable(xtcMonClient
    xtcMonClient.cc
)
target_link_libraries(xtcMonClient
    xtcdata::xtc
    monreq
    rt
)

add_executable(xtcMonWriter
    xtcMonWriter.cc
)
target_link_libraries(xtcMonWriter
    xtcdata::xtc
    monreq
    rt
)

add_executable(monReqServer
    monReqServer.cc
)
target_link_libraries(monReqServer
  xtcdata::xtc
  service
  collection
  eb
  monreq
  pthread
  rt
)

install(TARGETS monreq
  xtcMonClient
  xtcMonWriter
  monReqServer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
