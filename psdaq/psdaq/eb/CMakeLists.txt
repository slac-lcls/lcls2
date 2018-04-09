add_library(eb SHARED
  Endpoint.cc
  Batch.cc
  BatchManager.cc
  EventBuilder.cc
  EbEpoch.cc
  EbEvent.cc
  EbLfBase.cc
  EbLfServer.cc
  EbLfClient.cc
)

target_include_directories(eb PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  $ENV{CONDA_PREFIX}/include
)

find_library(FABRIC_CPP_LIBRARY fabric PATH $ENV{CONDA_PREFIX}/lib)

target_link_libraries(eb
  xtcdata
  service
  ${FABRIC_CPP_LIBRARY}
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

add_executable(tstEbContributor tstEbContributor.cc)
add_executable(tstEbBuilder     tstEbBuilder.cc)
add_executable(drpEbBuilder     drpEbBuilder.cc)

#target_compile_options(eb               PRIVATE -pg)
#target_compile_options(tstEbContributor PRIVATE -pg)
#target_compile_options(tstEbBuilder     PRIVATE -pg)

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
  ${FABRIC_CPP_LIBRARY}
)

target_link_libraries(tstEbContributor
  eb
  pthread
  rt
)

target_link_libraries(tstEbBuilder
  eb
  pthread
  rt
)

target_link_libraries(drpEbBuilder
  eb
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
