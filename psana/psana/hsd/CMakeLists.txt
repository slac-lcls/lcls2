cmake_minimum_required(VERSION 3.10)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Find the MyLibrary package
find_package(psalg REQUIRED)
find_package(xtcdata REQUIRED)


# Create the library target
add_library(psana_hsd SHARED
    hsd_digitizer_link.cc
)

# This includes the headers during the build and also installs the headers in the include dir for lcls2
target_include_directories(psana_hsd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

# Link the executable with the MyLibrary library
target_link_libraries(psana_hsd PUBLIC psalg::digitizer)




