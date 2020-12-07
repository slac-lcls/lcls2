find_package(PythonLibs REQUIRED)

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
    Json2Xtc.cc
)

target_include_directories(service PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${PYTHON_INCLUDE_DIRS}
    ${RAPIDJSON_INCLUDE_DIRS}
)

target_link_libraries(service PUBLIC
    xtcdata::xtc
    dl
    rt
    ${PYTHON_LIBRARIES}
)

add_library(collection SHARED
    Collection.cc
    kwargs.cc
)

target_link_libraries(collection
    psalg::utils
    libzmq
)

target_include_directories(collection PUBLIC
     $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
     $<INSTALL_INTERFACE:include>
)

add_library(exporter SHARED
    MetricExporter.cc
)

target_include_directories(exporter PUBLIC
     $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
     $<INSTALL_INTERFACE:include>
)

target_link_libraries(exporter PUBLIC
    psalg::utils
    prometheus-cpp::pull
)


install(TARGETS service collection exporter
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
