#install(FILES
#    AllocArray.hh
#    Allocator.hh
#    DESTINATION include/psalg/alloc
#)
##### New
cmake_minimum_required(VERSION 3.10)

# Create the library target
add_library(alloc_headers SHARED
            AllocArray.hh
            Allocator.hh
)

install(TARGETS alloc_headers
        EXPORT alloc_headersTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

# Generate and install the export file for the library
install(EXPORT alloc_headersTargets
    FILE alloc_headersTargets.cmake
    NAMESPACE alloc_headers::
    DESTINATION lib/cmake/alloc_headers
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/psalgConfig.cmake.in"
    ${CMAKE_CURRENT_BINARY_DIR}/psalgConfig.cmake
    INSTALL_DESTINATION lib/cmake/psalg
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/psalgConfig.cmake
    DESTINATION lib/cmake/psalg
)

# Generate the peaksConfig.cmake file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/alloc_headersConfig.cmake.in
    ${CMAKE_BINARY_DIR}/psana/peakFinder/peaksConfig.cmake
    INSTALL_DESTINATION lib/cmake/peaks
)

# Install the MyLibraryConfig.cmake file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/peaksConfig.cmake
    DESTINATION lib/cmake/peaks
)
