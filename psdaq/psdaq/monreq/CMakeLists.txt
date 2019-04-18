add_executable(monReqServer
    monReqServer.cc
)

target_link_libraries(monReqServer
    psalg::shmemsrv
    eventBuilder
    collection
    pthread
    rt
)

install(TARGETS
    monReqServer
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
