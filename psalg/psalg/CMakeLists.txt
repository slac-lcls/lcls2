install(FILES
    include/LocalExtrema.h
    include/PeakFinderAlgos.h
    include/Types.h
    include/stream.hh
    include/hsd.hh
    include/AllocArray.hh
    include/Allocator.hh
    include/ArrayIO.h
    include/Logger.h
    DESTINATION include/psalg/include
)

add_library(psalg SHARED
    src/LocalExtrema.cpp
    src/PeakFinderAlgos.cpp
    src/stream.cc
    src/ArrayIO.cpp
    src/hsd.cpp
)

target_include_directories(psalg PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/../install/include>
    $<INSTALL_INTERFACE:../install/include>
)

# datareader - test of data reader from non-xtc files
add_executable(datareader
    app/datareader.cc
)
target_link_libraries(datareader
    psalg
    xtcdata
)

add_executable(hsd_valid test/hsd_valid.cc)
target_include_directories(hsd_valid PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(hsd_valid
    psalg
    xtcdata
)

# Needed for valgrind
include(CTest) # ctest -T memcheck

# Test 1: AllocArray
add_executable(test_array
    test/test_array.cpp
)
target_link_libraries(test_array
    psalg
    xtcdata
)
add_test(NAME test_array COMMAND ${CMAKE_BINARY_DIR}/psalg/test_array
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Test 2: Peak finder
add_executable(test_peakFinder
    test/peakHeap.cpp
)
target_link_libraries(test_peakFinder
    psalg
    xtcdata
)
add_test(NAME test_peakFinder COMMAND ${CMAKE_BINARY_DIR}/psalg/test_peakFinder
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Test 3: Hsd
add_executable(test_hsd
    test/test_hsd.cpp
)
target_link_libraries(test_hsd
    psalg
    xtcdata
)
add_test(NAME test_hsd COMMAND ${CMAKE_BINARY_DIR}/psalg/test_hsd
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS psalg
                test_array
                test_peakFinder
                test_hsd
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
