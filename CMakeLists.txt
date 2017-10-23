cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

add_compile_options(-Wall)
add_compile_options(-Wno-uninitialized)

find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs 3.5 REQUIRED)

file(GLOB_RECURSE
     ALL_CXX_SOURCE_FILES
     *.cc *.hh
)

# Adding clang-format target if executable is found
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
  add_custom_target(
    clang-format
    COMMAND ${CLANG_FORMAT}
    -i
    -style=file
    ${ALL_CXX_SOURCE_FILES}
)
endif()


execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
    "import numpy; print(numpy.get_include(), end='')"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(HDF5 REQUIRED HL)

add_subdirectory(xtcdata)
add_subdirectory(psdaq)
add_subdirectory(pgp)
add_subdirectory(drp)
