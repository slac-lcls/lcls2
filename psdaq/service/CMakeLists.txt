
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
)

target_include_directories(service PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
#    $<INSTALL_INTERFACE:include>
)

#add_executable(h5writer
#    ../app/h5writer.cc
#)
#target_link_libraries(h5writer
#    pshdf5
#    xtc
#)
