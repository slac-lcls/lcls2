add_library(peaks SHARED
    src/LocalExtrema.cc
    src/PeakFinderAlgos.cc
    src/PeakFinderAlgosLCLS1.cc
    src/WFAlgos.cc
)

# calib included for NDArray
target_link_libraries(peaks
    xtcdata::xtc
    calib
)

target_include_directories(peaks PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
