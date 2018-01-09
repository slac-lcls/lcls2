
add_library(kcu SHARED
    Reg.cc
    GthEyeScan.cc
)

target_include_directories(kcu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)


#
#  xStatus
#
add_executable(kcuStatus xStatus.cpp)

add_executable(kcuLoopTest xLoopTest.cpp)

add_executable(kcuRead xRead.cpp)

target_link_libraries(kcuRead
   pthread
)

add_executable(kcuWrite xWrite.cpp)

target_link_libraries(kcuWrite
   pthread
)

add_executable(kcuDmaStatus xDmaStatus.cpp)

add_executable(kcuMargin xMargin.cpp)

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


