cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# cmake commands to possibly use in the future

#file(GLOB_RECURSE
#     ALL_CXX_SOURCE_FILES
#     *.cc *.hh
#)

# Adding clang-format target if executable is found
#find_program(CLANG_FORMAT "clang-format")
#if(CLANG_FORMAT)
#  add_custom_target(
#    clang-format
#    COMMAND ${CLANG_FORMAT}
#    -i
#    -style=file
#    ${ALL_CXX_SOURCE_FILES}
#)
#endif()

add_subdirectory(xtcdata)
