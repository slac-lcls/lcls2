add_library(service SHARED
    SysClk.cc
    Semaphore.cc
    SemLock.cc
    Lock.cc
    Pool.cc
    Task.cc
    TaskObject.cc
    Timer.cc
    GenericPool.cc
    GenericPoolW.cc
    Histogram.cc
    Dl.cc
)

target_include_directories(service PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(service PUBLIC
    dl
)

add_library(collection SHARED
    Collection.cc
)

target_link_libraries(collection
    libzmq
)

target_include_directories(collection PUBLIC
     $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
     $<INSTALL_INTERFACE:include>
)

add_library(exporter SHARED
    MetricExporter.cc
)

target_link_libraries(exporter PUBLIC
    prometheus-cpp::pull
)


install(TARGETS service collection exporter
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
