cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

find_package(xtcdata REQUIRED)
find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs 3.5 REQUIRED)
find_package(roentdek)

if (ROENTDEK_FOUND)
    add_subdirectory(psana/hexanode)
endif()
add_subdirectory(psana/peakFinder)
add_subdirectory(psana/constFracDiscrim)

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

# Test include "roentdek/resort64c.h"
if (ROENTDEK_FOUND)
    add_executable(ex01_resort64c
        psana/tests/ex01_resort64c.cc
    )
    target_link_libraries(ex01_resort64c
        peaks
        roentdek::resort64c
        #$ENV{CONDA_PREFIX}/lib/libResort64c_x64.a
        #/reg/g/psdm/sw/conda2/inst/envs/ps-2.0.3/lib/libResort64c_x64.a
    )

    ## Test ex02_read_lmf_data
    add_executable(ex02_read_lmf_data
        psana/tests/ex02_read_lmf_data.cc
    )
    target_link_libraries(ex02_read_lmf_data
        peaks
    )

    ## Test ex03_lmf2txt
    add_executable(ex03_lmf2txt
        psana/tests/ex03_lmf2txt.cc
    )
    target_link_libraries(ex03_lmf2txt
        peaks
    )

    ## Test ex04_sort
    add_executable(ex04_sort
        psana/tests/ex04_sort.cc
    )
    target_link_libraries(ex04_sort
        peaks
        roentdek::resort64c
    )
endif()

#Test 2: Peak finder
add_executable(test_peakFinder
    psana/tests/peakHeap.cc
)
target_link_libraries(test_peakFinder
    xtcdata::xtc
)

## test_WFAlgos
add_executable(test_WFAlgos
    psana/tests/test_WFAlgos.cc
)

add_test(NAME test_peakFinder COMMAND ${CMAKE_BINARY_DIR}/psana/test_peakFinder
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

