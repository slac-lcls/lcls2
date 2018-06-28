install(FILES
    include/LocalExtrema.hh
    include/PeakFinderAlgos.hh
    include/Types.hh
    include/stream.hh
    include/hsd.hh
    include/AllocArray.hh
    include/Allocator.hh
    include/ArrayIO.hh
    include/ArrayMetadata.hh
    include/Logger.hh
    DESTINATION include/psalg/include
)

add_library(psalg SHARED
    src/LocalExtrema.cc
    src/PeakFinderAlgos.cc
    src/stream.cc
    src/ArrayIO.cc
    src/Logger.cc
    src/hsd.cc
)

target_include_directories(psalg PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/../install/include>
    $<INSTALL_INTERFACE:../install/include>
)

# this is necessary only on RHEL6 where we need to pick up clock_gettime
target_link_libraries(psalg
    rt
)

# datareader - test of data reader from non-xtc files
add_executable(datareader
    app/datareader.cc
)
target_link_libraries(datareader
    psalg
    xtcdata::xtc
)

add_executable(hsd_valid test/hsd_valid.cc)
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
    test/test_array.cc
)
target_link_libraries(test_array
    psalg
    xtcdata::xtc
)
add_test(NAME test_array COMMAND ${CMAKE_BINARY_DIR}/psalg/test_array
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Test 2: Peak finder
add_executable(test_peakFinder
    test/peakHeap.cc
)
target_link_libraries(test_peakFinder
    psalg
    xtcdata::xtc
)
add_test(NAME test_peakFinder COMMAND ${CMAKE_BINARY_DIR}/psalg/test_peakFinder
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Test 3: Hsd
add_executable(test_hsd
    test/test_hsd.cc
)
target_link_libraries(test_hsd
    psalg
    xtcdata::xtc
)
add_test(NAME test_hsd COMMAND ${CMAKE_BINARY_DIR}/psalg/test_hsd
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Test Logger
add_executable(test_logger
    test/test_logger.cc
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
)
