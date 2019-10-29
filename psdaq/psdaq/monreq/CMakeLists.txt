add_executable(monReqServer
    monReqServer.cc
)

target_link_libraries(monReqServer
    psalg::shmemsrv
    psalg::utils
    eventBuilder
    collection
    exporter
    pthread
    rt
)

install(TARGETS
    monReqServer
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
