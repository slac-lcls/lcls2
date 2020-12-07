find_package(PythonLibs REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

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
  ${PYTHON_INCLUDE_DIRS}
  Threads::Threads
)

target_link_libraries(utilities
  xtcdata::xtc
  service
  libfabric::fabric
  ${PYTHON_LIBRARIES}
)

add_library(contributor SHARED
  EbCtrbInBase.cc
  TebContributor.cc
  MebContributor.cc
  BatchManager.cc
  Batch.cc
)

target_include_directories(contributor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(contributor
  utilities
)

add_library(eventBuilder SHARED
  EbAppBase.cc
  EventBuilder.cc
  EbEpoch.cc
  EbEvent.cc
  BatchManager.cc
  Batch.cc
)

target_include_directories(eventBuilder PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(eventBuilder
  utilities
)

add_executable(ft_pingpong ft_pingpong.cc fiTransport.cc)

add_executable(ft_publish         ft_publish.cc)
add_executable(ft_sub             ft_sub.cc)
add_executable(ft_push            ft_push.cc)
add_executable(ft_request         ft_request.cc)
add_executable(ft_server          ft_server.cc)
add_executable(ft_client          ft_client.cc)
add_executable(ft_writemsg_client ft_writemsg_client.cc)
add_executable(ft_writemsg_server ft_writemsg_server.cc)

add_executable(eblf_pingpong    eblf_pingpong.cc)

add_executable(tstEbLfLink      tstEbLfLink.cc)

#add_executable(ibMon            ibMon.cc)

add_executable(ctrb
  ctrb.cc
)

add_executable(teb
  teb.cc
)

target_include_directories(teb PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  ${PYTHON_INCLUDE_DIRS}
)

#add_executable(tstNthSet        tstNthSet.cc)
#target_compile_options(tstNthSet        PRIVATE -mbmi2)

#target_include_directories(ibMon PUBLIC
#  $<INSTALL_INTERFACE:include>
#  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
#  $ENV{CONDA_PREFIX}/include
#)

target_link_libraries(ft_pingpong
  utilities
)

target_link_libraries(ft_publish
  utilities
  Threads::Threads
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

target_link_libraries(eblf_pingpong
  utilities
  epicstools
)

target_link_libraries(tstEbLfLink
  utilities
  Threads::Threads
  rt
)

#target_link_libraries(ibMon
#  zmq
#  Threads::Threads
#  rt
#)
#

target_link_libraries(ctrb
  contributor
  collection
  exporter
  Threads::Threads
  rt
)

target_link_libraries(teb
  eventBuilder
  trigger
  collection
  exporter
  Threads::Threads
  rt
  dl
  xtcdata::xtc
  ${PYTHON_LIBRARIES}
)

install(TARGETS
  utilities
  contributor
  eventBuilder
  ctrb
  teb
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

add_executable(tstIndexPool    tstIndexPool.cc IndexPool.cc)

target_include_directories(tstIndexPool PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(tstIndexPool
)

#
# The following builds for use with gprof
#

#add_executable(ctrb
#  ctrb.cc
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
#)
#
#add_executable(teb
#  teb.cc
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
#)
#
#target_compile_options(ctrb PRIVATE -pg)
#target_compile_options(teb  PRIVATE -pg)
#
#target_link_libraries(ctrb
#  xtcdata::xtc
#  service
#  collection
#  fabric
#  zmq
#  Threads::Threads
#  rt
#  -pg
#)
#
#target_link_libraries(teb
#  xtcdata::xtc
#  service
#  collection
#  fabric
#  zmq
#  Threads::Threads
#  rt
#  -pg
#)
#
#install(TARGETS
#  utilities
#  contributor
#  eventBuilder
#  ctrb
#  teb
#  ARCHIVE DESTINATION lib
#  LIBRARY DESTINATION lib
#  RUNTIME DESTINATION bin
#)
