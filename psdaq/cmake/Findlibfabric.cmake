find_package(PkgConfig)
pkg_check_modules(PC_LIBFABRIC QUIET libfabric)

set(LIBFABRIC_VERSION ${PC_LIBFABRIC_VERSION})

find_path(LIBFABRIC_INCLUDE_DIR
    NAMES rdma/fabric.h
    PATHS ${PC_LIBFABRIC_INCLUDE_DIRS}
    PATH_SUFFIXES rdma
)

find_library(LIBFABRIC_LIBRARY NAMES fabric
             HINTS ${PC_LIBFABRIC_LIBDIR} ${PC_LIBFABRIC_LIBRARY_DIRS}
)

mark_as_advanced(LIBFABRIC_FOUND LIBFABRIC_INCLUDE_DIR LIBFABRIC_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libfabric DEFAULT_MSG
                                  LIBFABRIC_LIBRARY LIBFABRIC_INCLUDE_DIR)

if(LIBFABRIC_FOUND)
    set(LIBFABRIC_LIBRARIES ${LIBFABRIC_LIBRARY})
    set(LIBFABRIC_INCLUDE_DIRS ${LIBFABRIC_INCLUDE_DIR})

    if(NOT TARGET libfabric::fabric)
        add_library(libfabric::fabric UNKNOWN IMPORTED)
        set_target_properties(libfabric::fabric PROPERTIES
            IMPORTED_LOCATION ${LIBFABRIC_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${LIBFABRIC_INCLUDE_DIR}
        )
    endif()
endif()
