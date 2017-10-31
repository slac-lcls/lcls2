
add_library(dgram SHARED
    dgram.cc
)

target_link_libraries(dgram
    xtc
    ${PYTHON_LIBRARIES}
)

target_include_directories(dgram PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> 
    ${PYTHON_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIRS}
)

set_target_properties(dgram PROPERTIES PREFIX "")
