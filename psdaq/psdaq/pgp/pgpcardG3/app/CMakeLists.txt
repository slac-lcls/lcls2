#
#  xStatus
#
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(xStatus xStatus.cpp)

target_include_directories(xStatus PUBLIC
   ${RapidJSON_INCLUDE_DIRS}
)

target_link_libraries(xStatus
   utils
)

add_executable(xLoopTest xLoopTest.cpp)

target_link_libraries(xLoopTest
   Threads::Threads
   rt
)

add_executable(xRead xRead.cpp)

target_link_libraries(xRead
   Threads::Threads
   rt
)

add_executable(xMargin xMargin.cpp)

target_link_libraries(xMargin
   Threads::Threads
)

add_executable(xError xError.cpp)

target_link_libraries(xError
   Threads::Threads
)


