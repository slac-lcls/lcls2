
add_library(kcu SHARED
    Reg.cc
    GthEyeScan.cc
    Client.cc
    Si570.cc
)

target_include_directories(kcu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ../../aes-stream-drivers/include
)

target_link_libraries(kcu
    xtcdata::xtc
)


#
#  xStatus
#
add_executable(kcuStatus xStatus.cpp)

target_include_directories(kcuStatus PUBLIC
    ../../aes-stream-drivers/include
)

target_link_libraries(kcuStatus 
    kcu
)

add_executable(kcuHist xHist.cpp)

target_include_directories(kcuHist PUBLIC
    ../../aes-stream-drivers/include
)

add_executable(kcuSim xSim.cpp)

target_include_directories(kcuSim PUBLIC
    ../../aes-stream-drivers/include
)

#add_executable(kcuSimProxy xSimProxy.cpp)

target_include_directories(kcuSimProxy PUBLIC
    ../../aes-stream-drivers/include
)

add_executable(kcuSimValid xSimValid.cpp)

target_include_directories(kcuSimValid PUBLIC
    ../../aes-stream-drivers/include
)

add_executable(kcuLoopTest xLoopTest.cpp)

target_include_directories(kcuLoopTest PUBLIC
    ../../aes-stream-drivers/include
)

add_executable(kcuReg xReg.cpp)

target_include_directories(kcuReg PUBLIC
    ../../aes-stream-drivers/include
)

add_executable(kcuRead xRead.cpp)

target_include_directories(kcuRead PUBLIC
    ../../aes-stream-drivers/include
)

target_link_libraries(kcuRead
   pthread
   rt
)

add_executable(kcuWrite xWrite.cpp)

target_include_directories(kcuWrite PUBLIC
    ../../aes-stream-drivers/include
)

target_link_libraries(kcuWrite
   pthread
   rt
)

add_executable(kcuDmaStatus xDmaStatus.cpp)

target_include_directories(kcuDmaStatus PUBLIC
    ../../aes-stream-drivers/include
)

add_executable(kcuMargin xMargin.cpp)

target_include_directories(kcuMargin PUBLIC
    ../../aes-stream-drivers/include
)

target_link_libraries(kcuMargin
   kcu
   pthread
)
#
#add_executable(xError xError.cpp)
#
#target_link_libraries(xError
#   pthread
#)


install(TARGETS kcu
                kcuSim
                kcuSimValid
                kcuHist
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
