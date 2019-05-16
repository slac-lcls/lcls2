find_package(prometheus-cpp REQUIRED)

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
)

add_library(collection SHARED
    Collection.cc
)

target_link_libraries(collection
    zmq
)

target_include_directories(collection PUBLIC
     $ENV{CONDA_PREFIX}/include
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
