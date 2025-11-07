
#
#  xStatus
#
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(qStatus qStatus.cc)
target_include_directories(qStatus PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../..>
)

add_executable(qConfig qConfig.cc)
target_include_directories(qConfig PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../..>
)


add_executable(qDesc   qDesc.cc)
target_include_directories(qDesc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../..>
)

add_executable(qRead   qRead.cc)
target_include_directories(qRead PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../..>
)

target_link_libraries(qRead
  dl
  Threads::Threads
  rt
)

add_executable(qReset   qReset.cc)
target_include_directories(qReset PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../..>
)

target_link_libraries(qRead
  dl
  Threads::Threads
  rt
)

#add_executable(qLoopTest qLoopTest.cc)
#target_include_directories(qLoopTest PUBLIC
#    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../..>
#    $ENV{CONDA_PREFIX}/include
#)

add_executable(qHist qHist.cc)
target_include_directories(qHist PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../..>
)

add_executable(cStatus cStatus.cc)
target_include_directories(cStatus PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../..>
)

add_executable(cLoopTest cLoopTest.cc)
target_include_directories(cLoopTest PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../..>
)

#install(TARGETS qStatus
#                qConfig
#                qRead
#                qLoopTest
#                qHist
#    RUNTIME DESTINATION bin
#)
