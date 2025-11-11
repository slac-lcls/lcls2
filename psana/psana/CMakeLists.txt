cmake_minimum_required(VERSION 3.10)

add_library(psana SHARED
    peakFinder/src/LocalExtrema.cc
    peakFinder/src/PeakFinderAlgos.cc
    peakFinder/src/PeakFinderAlgosLCLS1.cc
    peakFinder/src/WFAlgos.cc
)

# This includes the headers during the build and also installs the headers in the include dir for lcls2
target_include_directories(psana PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/peakFinder/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/hsd/include>
)

# Link the executable
target_link_libraries(psana PUBLIC psalg)
target_link_libraries(psana PUBLIC xtc)

add_custom_command(TARGET psana
    POST_BUILD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    COMMAND echo "PIP INSTALLING PSANA"
    COMMAND ${CMAKE_COMMAND} -E env
        INSTDIR=${CMAKE_INSTALL_PREFIX}
        PSALGDIR=${CMAKE_BINARY_DIR}/psalg/psalg
        XTCDATADIR=${CMAKE_BINARY_DIR}/xtcdata/xtcdata/xtc
        pip install --no-deps --prefix=${CMAKE_INSTALL_PREFIX} ${PIP_OPTIONS} .
)

