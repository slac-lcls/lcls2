cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

add_compile_options(-Wall)
add_compile_options(-Wno-uninitialized)
add_compile_options(-fno-omit-frame-pointer)

find_package(xtc REQUIRED)

option(PSANA_USE_LEGION "Enable support for Legion" OFF)
if(PSANA_USE_LEGION)
  find_package(Legion REQUIRED)
endif()

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


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(psdaq)
add_subdirectory(pgp)
add_subdirectory(drp)
