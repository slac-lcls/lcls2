set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(mmhw SHARED
    AxiVersion.cc
    HexFile.cc
    McsFile.cc
    Reg.cc
    Reg64.cc
    RegProxy.cc
    RingBuffer.cc
    GthEyeScan.cc
    TprCore.cc
    TriggerEventManager.cc
    Xvc.cc
)

target_include_directories(mmhw PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)

target_link_libraries(mmhw
    service
)

add_executable(mmregtest mmregtest.cc)

target_include_directories(mmregtest PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)

add_executable(mmxvc mmxvc.cc)

target_include_directories(mmxvc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)

target_link_libraries(mmxvc
    service
    mmhw
    Threads::Threads
    rt
)

install(TARGETS mmhw
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
