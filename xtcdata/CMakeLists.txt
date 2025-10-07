cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(xtcdata)

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

include(CMakePackageConfigHelpers)

install(EXPORT xtcdataTargets
    FILE xtcdataTargets.cmake
    NAMESPACE xtcdata::
    DESTINATION lib/cmake/xtcdata
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/xtcdataConfig.cmake.in"
    ${CMAKE_CURRENT_BINARY_DIR}/xtcdataConfig.cmake
    INSTALL_DESTINATION lib/cmake/xtcdata
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/xtcdataConfig.cmake
    DESTINATION lib/cmake/xtcdata
)
