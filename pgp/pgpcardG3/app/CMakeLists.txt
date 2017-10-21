
#
#  xStatus
#
add_executable(xStatus xStatus.cpp)

add_executable(xLoopTest xLoopTest.cpp)

add_executable(xRead xRead.cpp)

target_link_libraries(xRead
   pthread
)

