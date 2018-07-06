
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

install(TARGETS service collection
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
