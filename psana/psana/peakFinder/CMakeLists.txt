cmake_minimum_required(VERSION 3.10)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Find the MyLibrary package
find_package(psalg REQUIRED)
find_package(xtcdata REQUIRED)


# Create the library target
add_library(peaks SHARED
    #src/LocalExtrema.cc
    #src/PeakFinderAlgos.cc
    #src/PeakFinderAlgosLCLS1.cc
    src/WFAlgos.cc
)

# This includes the headers during the build and also installs the headers in the include dir for lcls2
target_include_directories(peaks PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link the executable with the MyLibrary library
target_link_libraries(peaks PUBLIC psalg::alloc)

add_custom_command(TARGET peaks
                   POST_BUILD
                   COMMAND echo "Pip install")


# This marks which targets are going to be exported and also defines instructions on
# where libraries will be installed.
install(TARGETS peaks
        EXPORT peaksTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

# Export the include directory
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include/psana/peakFinder)

# This will create an export set that includes the targets. Export sets are used
# to generate cmake files other projects can import and link against. Recall that install
# (cmake in generaly really) is just creating instructions. Here it is creating install
# instructions.
install(EXPORT peaksTargets
    FILE peaksTargets.cmake
    NAMESPACE peaks::
    DESTINATION lib/cmake/peaks
)

# Generate the peaksConfig.cmake file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/peaksConfig.cmake.in
    ${CMAKE_BINARY_DIR}/psana/peakFinder/peaksConfig.cmake
    INSTALL_DESTINATION lib/cmake/peaks
)

# Install the MyLibraryConfig.cmake file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/peaksConfig.cmake
    DESTINATION lib/cmake/peaks
)