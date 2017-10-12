
add_library(eb SHARED
  Endpoint.cc
  fiTransport.cc
)

target_include_directories(eb PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    /reg/common/package/libfabric/1.4.2/include
)

target_link_libraries(eb
    service
    /reg/common/package/libfabric/1.4.2/lib/libfabric.so
)

add_executable(ft_client ft_client.cc)
add_executable(ft_pingpong ft_pingpong.cc)
add_executable(ft_publish ft_publish.cc)
add_executable(ft_push ft_push.cc)
add_executable(ft_request ft_request.cc)
add_executable(ft_server ft_server.cc)
add_executable(ft_sub ft_sub.cc)

target_link_libraries(ft_client 
    eb
)

target_link_libraries(ft_pingpong 
    eb
)

target_link_libraries(ft_publish 
    eb
)

target_link_libraries(ft_push 
    eb
)

target_link_libraries(ft_request 
    eb
)

target_link_libraries(ft_server 
    eb
)

target_link_libraries(ft_sub 
    eb
)
