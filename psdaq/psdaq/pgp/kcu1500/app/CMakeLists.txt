set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(kcu SHARED
    Reg.cc
    GthEyeScan.cc
    Client.cc
    Si570.cc
)

target_include_directories(kcu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:include>
    ${RapidJSON_INCLUDE_DIRS}
)

target_link_libraries(kcu
    xtc
    utils
    mmhw
)

# A static version of the kcu library is needed for kcuStatus
add_library(staticKcu STATIC
    Reg.cc
    GthEyeScan.cc
    Client.cc
    Si570.cc
)

target_include_directories(staticKcu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(staticKcu
    staticXtc
)


#
#  xStatus
#
add_executable(kcuStatus xStatus.cpp)

target_include_directories(kcuStatus PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)

# kcuStatus needs to be runnable at boot time without the standard environment
target_link_libraries(kcuStatus
    staticKcu
)

add_executable(kcuPgpStatus xPgpStatus.cpp)

target_include_directories(kcuPgpStatus PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)

target_link_libraries(kcuPgpStatus
    kcu
)

add_executable(kcuHist xHist.cpp)

target_include_directories(kcuHist PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)

add_executable(kcuSim xSim.cpp)

target_include_directories(kcuSim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)

#add_executable(kcuSimProxy xSimProxy.cpp)

add_executable(kcuSimValid xSimValid.cpp)

target_include_directories(kcuSimValid PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:include>
)

add_executable(kcuLoopTest xLoopTest.cpp)

target_include_directories(kcuLoopTest PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)

add_executable(kcuReg xReg.cpp)

target_include_directories(kcuReg PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)

add_executable(kcuRead xRead.cpp)

target_include_directories(kcuRead PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(kcuRead
   Threads::Threads
   rt
)

add_executable(kcuReadBulk xReadBulk.cpp)

target_include_directories(kcuReadBulk PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(kcuReadBulk
   Threads::Threads
   rt
)

add_executable(kcuWrite xWrite.cpp)

target_include_directories(kcuWrite PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(kcuWrite
   Threads::Threads
   rt
)

add_executable(kcuDmaStatus xDmaStatus.cpp)

target_include_directories(kcuDmaStatus PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)

add_executable(kcuMargin xMargin.cpp)

target_include_directories(kcuMargin PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)

target_link_libraries(kcuMargin
   kcu
   Threads::Threads
)
#
#add_executable(xError xError.cpp)
#
#target_link_libraries(xError
#   Threads::Threads
#)


install(TARGETS kcu
                kcuSim
                kcuStatus
                kcuSimValid
                kcuHist
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
