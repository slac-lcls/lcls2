cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0")
    message(FATAL_ERROR "Insufficient gcc version")
  endif()
endif()

#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;$ENV{CONDA_PREFIX}/lib")
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#link_directories($ENV{CONDA_PREFIX}/lib)

add_subdirectory(utils)
add_subdirectory(calib)
add_subdirectory(alloc)
add_subdirectory(peaks)
add_subdirectory(digitizer)
#add_subdirectory(detector)

add_library(psalg SHARED
    calib/src/ArrayIO.cc
###   utils/src/Logger.cc
)

target_include_directories(psalg PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/../install/include>
    $<INSTALL_INTERFACE:../install/include>
)

# this is necessary only on RHEL6 where we need to pick up clock_gettime
target_link_libraries(psalg
    utils
    calib
    peaks
    digitizer
    rt
)
    #detector

# datareader - test of data reader from non-xtc files
add_executable(datareader
    app/datareader.cc
)
target_link_libraries(datareader
    psalg
    xtcdata::xtc
)

add_executable(hsd_valid tests/hsd_valid.cc)
target_include_directories(hsd_valid PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(hsd_valid
    psalg
    xtcdata::xtc
)

# Needed for valgrind
include(CTest) # ctest -T memcheck

# Test 1: AllocArray
add_executable(test_array
    tests/test_array.cc
)
target_link_libraries(test_array
    psalg
    xtcdata::xtc
)
add_test(NAME test_array COMMAND ${CMAKE_BINARY_DIR}/psalg/test_array
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Test 2: Peak finder
add_executable(test_peakFinder
    tests/peakHeap.cc
)
target_link_libraries(test_peakFinder
    psalg
    xtcdata::xtc
)
add_test(NAME test_peakFinder COMMAND ${CMAKE_BINARY_DIR}/psalg/test_peakFinder
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Test 3: Hsd
add_executable(test_hsd
    tests/test_hsd.cc
)
target_link_libraries(test_hsd
    psalg
    xtcdata::xtc
)
add_test(NAME test_hsd COMMAND ${CMAKE_BINARY_DIR}/psalg/test_hsd
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Test Logger
add_executable(test_logger
    tests/test_logger.cc
)
target_link_libraries(test_logger
    psalg
    xtcdata::xtc
)

install(TARGETS psalg
                test_array
                test_peakFinder
                test_hsd
                test_logger
                datareader
    EXPORT psalgTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(EXPORT psalgTargets
    FILE psalgConfig.cmake
    DESTINATION lib/cmake/psalg
    NAMESPACE psalg::
)
