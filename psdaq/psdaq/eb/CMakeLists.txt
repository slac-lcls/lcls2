add_library(utilities SHARED
  utilities.cc
  IndexPool.cc
  Endpoint.cc
  EbLfLink.cc
  EbLfServer.cc
  EbLfClient.cc
)

target_include_directories(utilities PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  $ENV{CONDA_PREFIX}/include
)

target_link_libraries(utilities
  xtcdata::xtc
  service
  fabric
)

add_library(ctrb SHARED
  EbCtrbInBase.cc
  TebContributor.cc
  MebContributor.cc
  BatchManager.cc
  Batch.cc
  StatsMonitor.cc
)

target_include_directories(ctrb PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  $ENV{CONDA_PREFIX}/include
)

target_link_libraries(ctrb
  utilities
)

add_library(eb SHARED
  EbAppBase.cc
  EventBuilder.cc
  EbEpoch.cc
  EbEvent.cc
  BatchManager.cc
  Batch.cc
  StatsMonitor.cc
)

target_include_directories(eb PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  $ENV{CONDA_PREFIX}/include
)

target_link_libraries(eb
  utilities
)

#add_executable(ft_pingpong ft_pingpong.cc fiTransport.cc)

add_executable(ft_publish         ft_publish.cc)
add_executable(ft_sub             ft_sub.cc)
add_executable(ft_push            ft_push.cc)
add_executable(ft_request         ft_request.cc)
add_executable(ft_server          ft_server.cc)
add_executable(ft_client          ft_client.cc)
add_executable(ft_writemsg_client ft_writemsg_client.cc)
add_executable(ft_writemsg_server ft_writemsg_server.cc)

#add_executable(eblf_pingpong    eblf_pingpong.cc)
#
#add_executable(tstEbLfLink      tstEbLfLink.cc)
#
#add_executable(ibMon            ibMon.cc)
#
add_executable(tstEbContributor
  tstEbContributor.cc
)
add_executable(tstEbBuilder
  tstEbBuilder.cc
)
#add_executable(drpEbBuilder
#  drpEbBuilder.cc
#)
#
#add_executable(tstNthSet        tstNthSet.cc)
#target_compile_options(tstNthSet        PRIVATE -mbmi2)

#target_include_directories(ibMon PUBLIC
#  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
#  $<INSTALL_INTERFACE:include>
#  $ENV{CONDA_PREFIX}/include
#)

#target_link_libraries(ft_pingpong
#  utilities
#)

target_link_libraries(ft_publish
  utilities
  pthread
)

target_link_libraries(ft_sub
  utilities
)

target_link_libraries(ft_push
  utilities
)

target_link_libraries(ft_request
  utilities
)

target_link_libraries(ft_client
  utilities
)

target_link_libraries(ft_server
  utilities
)

target_link_libraries(ft_writemsg_client
  utilities
)

target_link_libraries(ft_writemsg_server
  utilities
)

#target_link_libraries(eblf_pingpong
#  utilities
#)
#
#target_link_libraries(tstEbLfLink
#  utilities
#  pthread
#  rt
#)
#
#target_link_libraries(ibMon
#  zmq
#  pthread
#  rt
#)
#
target_link_libraries(tstEbContributor
  ctrb
  collection
  pthread
  rt
)

target_link_libraries(tstEbBuilder
  eb
  collection
  pthread
  rt
)

#target_link_libraries(drpEbBuilder
#  eb
#  collection
#  pthread
#  rt
#)

install(TARGETS utilities
  tstEbContributor
  tstEbBuilder
  #drpEbBuilder
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

add_executable(tstIndexPool    tstIndexPool.cc IndexPool.cc)

target_include_directories(tstIndexPool PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  $ENV{CONDA_PREFIX}/include
)

target_link_libraries(tstIndexPool
)

#
# The following builds for use with gprof
#

#add_executable(tstEbContributor tstEbContributor.cc
#  utilities.cc
#  Endpoint.cc
#  IndexPool.cc
#  Batch.cc
#  BatchManager.cc
#  EventBuilder.cc
#  EbEpoch.cc
#  EbEvent.cc
#  EbLfLink.cc
#  EbLfServer.cc
#  EbLfClient.cc
#  TebContributor.cc
#  EbCtrbInBase.cc
#  EbAppBase.cc
#  MebContributor.cc
#  StatsMonitor.cc
#)
#
#add_executable(tstEbBuilder     tstEbBuilder.cc
#  utilities.cc
#  Endpoint.cc
#  IndexPool.cc
#  Batch.cc
#  BatchManager.cc
#  EventBuilder.cc
#  EbEpoch.cc
#  EbEvent.cc
#  EbLfLink.cc
#  EbLfServer.cc
#  EbLfClient.cc
#  TebContributor.cc
#  EbCtrbInBase.cc
#  EbAppBase.cc
#  MebContributor.cc
#  StatsMonitor.cc
#)
#
#target_compile_options(tstEbContributor PRIVATE -pg)
#target_compile_options(tstEbBuilder     PRIVATE -pg)
#
#target_link_libraries(tstEbContributor
#  xtcdata::xtc
#  service
#  collection
#  fabric
#  zmq
#  pthread
#  rt
#  -pg
#)
#
#target_link_libraries(tstEbBuilder
#  xtcdata::xtc
#  service
#  collection
#  fabric
#  zmq
#  pthread
#  rt
#  -pg
#)
#
#install(TARGETS eb
#  tstEbContributor
#  tstEbBuilder
#  #drpEbBuilder
#  ARCHIVE DESTINATION lib
#  LIBRARY DESTINATION lib
#  RUNTIME DESTINATION bin
#)
