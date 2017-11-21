
#
#  xStatus
#
add_executable(xStatus xStatus.cpp)

add_executable(xLoopTest xLoopTest.cpp)

add_executable(xRead xRead.cpp)

target_link_libraries(xRead
   pthread
   rt
)

add_executable(xMargin xMargin.cpp)

target_link_libraries(xMargin
   pthread
   rt
)

add_executable(xError xError.cpp)

target_link_libraries(xError
   pthread
   rt
)


