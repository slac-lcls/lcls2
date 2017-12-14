
#
#  xStatus
#
add_executable(kcuStatus xStatus.cpp)

#add_executable(xLoopTest xLoopTest.cpp)
#
add_executable(kcuRead xRead.cpp)

target_link_libraries(kcuRead
   pthread
)

add_executable(kcuWrite xWrite.cpp)

add_executable(kcuDmaStatus xDmaStatus.cpp)

#
#add_executable(xMargin xMargin.cpp)
#
#target_link_libraries(xMargin
#   pthread
#)
#
#add_executable(xError xError.cpp)
#
#target_link_libraries(xError
#   pthread
#)


