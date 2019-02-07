cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

add_compile_options(-Wall)
add_compile_options(-Wno-uninitialized)
add_compile_options(-fno-omit-frame-pointer)

find_package(xtcdata REQUIRED)
find_package(psalg REQUIRED)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0")
    message(FATAL_ERROR "Insufficient gcc version")
  endif()
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;$ENV{CONDA_PREFIX}/lib")
# We are explicitly added the conda lib to the RPATH and stopping others paths from
# being added implicitly by link_directories
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

link_directories($ENV{CONDA_PREFIX}/lib)

add_subdirectory(psdaq)
add_subdirectory(drp)
