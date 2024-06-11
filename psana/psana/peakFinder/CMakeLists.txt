cmake_minimum_required(VERSION 3.10)

# Find the MyLibrary package
# find_package(psalg REQUIRED)
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
target_link_libraries(peaks PUBLIC xtcdata)

add_custom_command(TARGET peaks
                   POST_BUILD
                   COMMAND echo "Pip install")


# Export the library
# TARGETS: specifies which targets
# EXPORT: This creates an export set named peaksTargets that
# includes the target "peaks". Export sets are used to generate cmake files
# other projects can use to import and link against said target.
install(TARGETS peaks
        EXPORT peaksTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

# Export the include directory
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include/psana/peakFinder)

# Generate and install the export file for the library
install(EXPORT peaksTargets
    FILE peaksTargets.cmake
    NAMESPACE peaks::
    DESTINATION lib/cmake/peaks
)

# Generate the peaksConfig.cmake file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/peaksConfig.cmake.in
    ${CMAKE_BINARY_DIR}/peaksConfig.cmake
    INSTALL_DESTINATION lib/cmake/peaks
)

# Install the MyLibraryConfig.cmake file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/peaksConfig.cmake
    DESTINATION lib/cmake/peaks
)
