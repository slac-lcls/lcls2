
#
#  xStatus
#
add_executable(qStatus qStatus.cc)
target_include_directories(qStatus PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)

add_executable(qConfig qConfig.cc)
target_include_directories(qConfig PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)


add_executable(qRead   qRead.cc)
target_include_directories(qRead PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)

target_link_libraries(qRead
  dl
  pthread
  rt
)

add_executable(qReset   qReset.cc)
target_include_directories(qReset PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)

target_link_libraries(qRead
  dl
  pthread
  rt
)

add_executable(qLoopTest qLoopTest.cc)
target_include_directories(qLoopTest PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)

add_executable(qHist qHist.cc)
target_include_directories(qHist PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)

#install(TARGETS qStatus
#                qConfig
#                qRead
#                qLoopTest
#                qHist
#    RUNTIME DESTINATION bin
#)
