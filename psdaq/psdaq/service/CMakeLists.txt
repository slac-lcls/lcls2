find_package(PythonLibs REQUIRED)
find_package(nlohmann_json REQUIRED)

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
    IpcUtils.cc
)

target_include_directories(service PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:include>
    ${PYTHON_INCLUDE_DIRS}
    ${RapidJSON_INCLUDE_DIRS}
    ${XTCDATA_INCLUDE_DIRS}
)

target_link_libraries(service PUBLIC
    xtc
    utils
    dl
    rt
    ${PYTHON_LIBRARIES}
)

add_library(collection SHARED
    Collection.cc
    kwargs.cc
    range.cc
)

target_link_libraries(collection
    utils
    libzmq
    nlohmann_json::nlohmann_json
)

target_include_directories(collection PUBLIC
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
     $<INSTALL_INTERFACE:include>
     ${ZeroMQ_INCLUDE_DIRS}
)

add_library(exporter SHARED
    MetricExporter.cc
)

target_include_directories(exporter PUBLIC
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
     $<INSTALL_INTERFACE:include>
)

target_link_libraries(exporter PUBLIC
    utils
    prometheus-cpp::pull
    prometheus-cpp::push
)

install(FILES
    EbDgram.hh
    DESTINATION include/psdaq/service
)

install(TARGETS service collection exporter
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
