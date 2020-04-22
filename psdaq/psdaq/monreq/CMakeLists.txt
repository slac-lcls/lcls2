set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(monReqServer
    monReqServer.cc
)

target_link_libraries(monReqServer
    psalg::shmemsrv
    psalg::utils
    eventBuilder
    collection
    exporter
    Threads::Threads
    rt
)

install(TARGETS
    monReqServer
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
