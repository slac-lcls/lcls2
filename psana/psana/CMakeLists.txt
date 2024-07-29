cmake_minimum_required(VERSION 3.10)

# Find the MyLibrary package
find_package(psalg REQUIRED)
find_package(xtcdata REQUIRED)

add_library(psana SHARED
    peakFinder/src/LocalExtrema.cc
    peakFinder/src/PeakFinderAlgos.cc
    peakFinder/src/PeakFinderAlgosLCLS1.cc
    peakFinder/src/WFAlgos.cc
    PSALG_LINK.cc)
    

# This includes the headers during the build and also installs the headers in the include dir for lcls2
target_include_directories(psana PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/peakFinder>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/hsd>
)

# Link the executable with the MyLibrary library
target_link_libraries(psana PUBLIC psalg::alloc)
target_link_libraries(psana PUBLIC psalg::digitizer)

add_custom_command(TARGET psana
                   POST_BUILD
                   WORKING_DIRECTORY ${PSANA_PATH}
                   COMMAND echo "BUILDING EXTENSIONS"
                   COMMAND python setup.py build_ext -f --inplace
                   COMMAND echo "PIP INSTALLING PSANA"
                   COMMAND pip install --no-deps --prefix=${CMAKE_INSTALL_PREFIX} -e .)