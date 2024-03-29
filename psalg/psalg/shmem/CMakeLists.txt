add_library(shmemsrv SHARED
  XtcMonitorServer.cc
  XtcMonitorMsg.cc
  TransitionCache.cc
  ProcInfo.cc
  XtcRunSet.cc
)

add_library(shmemcli SHARED
  ShmemClient.cc
  XtcMonitorMsg.cc
)

target_include_directories(shmemcli PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(shmemsrv PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(shmemcli
  xtcdata::xtc
  rt
)

target_link_libraries(shmemsrv
  xtcdata::xtc
)

add_executable(shmemClient
  shmemClientApp.cc
)

target_link_libraries(shmemClient
  xtcdata::xtc
  shmemcli
  rt
)

add_executable(shmemWriter
  shmemWriter.cc
)

target_link_libraries(shmemWriter
  xtcdata::xtc
  shmemcli
  rt
)

add_executable(shmemServer
  shmemServer.cc
)

target_link_libraries(shmemServer
  xtcdata::xtc
  shmemsrv
  rt
)

target_include_directories(shmemcli PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

install(FILES
  ShmemClient.hh
  XtcMonitorServer.hh
  XtcMonitorMsg.hh
  DESTINATION include/psalg/shmem
)

install(TARGETS shmemcli
  shmemClient shmemWriter
  EXPORT psalgTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS shmemsrv
  shmemServer
  EXPORT psalgTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
