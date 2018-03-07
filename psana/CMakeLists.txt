cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

find_package(xtcdata REQUIRED)
find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs 3.5 REQUIRED)

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
    "import numpy; print(numpy.get_include(), end='')"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS
)

add_library(dgram SHARED
    src/dgram.cc
)

target_link_libraries(dgram
    xtcdata
    ${PYTHON_LIBRARIES}
)

target_include_directories(dgram PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> 
    ${PYTHON_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIRS}
)

set_target_properties(dgram PROPERTIES PREFIX "")

