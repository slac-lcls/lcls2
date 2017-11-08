
add_library(eb SHARED
  Endpoint.cc
  Batch.cc
  BatchManager.cc
  EventBuilder.cc
  EbEpoch.cc
  EbEvent.cc
  EbFtBase.cc
  EbFtServer.cc
  EbFtClient.cc
)

target_include_directories(eb PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    /reg/common/package/libfabric/1.5.1/include
)

target_link_libraries(eb
    xtc
    service
    /reg/common/package/libfabric/1.5.1/lib/libfabric.so
)

add_executable(ft_pingpong ft_pingpong.cc fiTransport.cc)

add_executable(ft_publish ft_publish.cc)
add_executable(ft_sub ft_sub.cc)
add_executable(ft_push ft_push.cc)
add_executable(ft_request ft_request.cc)
add_executable(ft_server ft_server.cc)
add_executable(ft_client ft_client.cc)
add_executable(ft_writemsg_client ft_writemsg_client.cc)
add_executable(ft_writemsg_server ft_writemsg_server.cc)

add_executable(tstEbContributor tstEbContributor)
add_executable(tstEbBuilder tstEbBuilder)

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

target_link_libraries(tstEbContributor
    eb
    pthread
)

target_link_libraries(tstEbBuilder
    eb
    pthread
)
