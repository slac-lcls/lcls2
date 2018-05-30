add_library(psalg SHARED
    src/LocalExtrema.cpp
    src/PeakFinderAlgos.cpp
)

target_include_directories(psalg PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
    ../../install/include
)

install(TARGETS psalg
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES
    include/LocalExtrema.h
    include/PeakFinderAlgos.h
    include/Types.h
    DESTINATION include/psalg
)

# Needed for valgrind
include(CTest) # ctest -T memcheck

# Test 1: AllocArray
add_executable(test_array
    test/test_array.cpp
)
target_link_libraries(test_array
    psalg
)
add_test(NAME test_array COMMAND ${CMAKE_BINARY_DIR}/psalg/test_array
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Test 2: Peak finder
add_executable(test_peakFinder
    test/peakHeap.cpp
)
target_link_libraries(test_peakFinder
    psalg
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