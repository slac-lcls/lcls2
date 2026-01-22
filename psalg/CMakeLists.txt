project(psalg)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(RapidJSON REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenMP REQUIRED)
# we don't have this package for all OS's, so make it optional
find_package(roentdek)

enable_testing()

add_subdirectory(psalg)

include(CMakePackageConfigHelpers)

install(EXPORT psalgTargets
    FILE psalgTargets.cmake
    NAMESPACE psalg::
    DESTINATION lib/cmake/psalg
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/psalgConfig.cmake.in"
    ${CMAKE_CURRENT_BINARY_DIR}/psalgConfig.cmake
    INSTALL_DESTINATION lib/cmake/psalg
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/psalgConfig.cmake
    DESTINATION lib/cmake/psalg
)


