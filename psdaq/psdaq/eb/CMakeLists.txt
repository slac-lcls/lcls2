add_library(eb SHARED
  utilities.cc
  Endpoint.cc
  Batch.cc
  BatchManager.cc
  EventBuilder.cc
  EbEpoch.cc
  EbEvent.cc
  EbLfLink.cc
  EbLfServer.cc
  EbLfClient.cc
  EbContributor.cc
  EbCtrbInBase.cc
  EbAppBase.cc
  MonContributor.cc
  StatsMonitor.cc
)

target_include_directories(eb PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  $ENV{CONDA_PREFIX}/include
)

target_link_libraries(eb
  xtcdata::xtc
  service
  fabric
  zmq
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

add_executable(ibMon            ibMon.cc)

add_executable(tstEbContributor tstEbContributor.cc)
#add_executable(drpEbContributor drpEbContributor.cc)
add_executable(tstEbBuilder     tstEbBuilder.cc)
add_executable(drpEbBuilder     drpEbBuilder.cc)

add_executable(tstNthSet        tstNthSet.cc)
target_compile_options(tstNthSet        PRIVATE -mbmi2)

#target_compile_options(eb               PRIVATE -pg)
#target_compile_options(tstEbContributor PRIVATE -pg)
#target_compile_options(tstEbBuilder     PRIVATE -pg)

target_include_directories(ibMon PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  $ENV{CONDA_PREFIX}/include
)

target_link_libraries(ft_pingpong
  eb
)

target_link_libraries(ft_publish
  eb
  pthread
)

target_link_libraries(ft_sub
  eb
)

target_link_libraries(ft_push
  eb
)

target_link_libraries(ft_request
  eb
)

target_link_libraries(ft_client
  eb
)

target_link_libraries(ft_server
  eb
)

target_link_libraries(ft_writemsg_client
  eb
)

target_link_libraries(ft_writemsg_server
  eb
)

target_link_libraries(eblf_pingpong
  eb
)

target_link_libraries(tstEbLfLink
  eb
  pthread
  rt
)

target_link_libraries(ibMon
  zmq
  pthread
  rt
)

target_link_libraries(tstEbContributor
  eb
  pthread
  rt
)

#target_link_libraries(drpEbContributor
#  eb
#  pthread
#  rt
#)

target_link_libraries(tstEbBuilder
  eb
  pthread
  rt
)

target_link_libraries(drpEbBuilder
  eb
  collection
  pthread
  rt
)

install(TARGETS eb
  ft_pingpong
  ft_publish
  ft_sub
  ft_push
  ft_request
  ft_client
  ft_server
  ft_writemsg_client
	ft_writemsg_server
  eblf_pingpong
  tstEbContributor
	tstEbBuilder
	drpEbBuilder
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
