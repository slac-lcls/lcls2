add_library(psalg SHARED
    src/LocalExtrema.cpp
    src/PeakFinderAlgos.cpp
)

target_include_directories(psalg PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
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