find_path(CPSW_INCLUDE_DIR
    NAMES cpsw_error.h
)

find_library(CPSW_LIBRARY NAMES cpsw
)

mark_as_advanced(CPSW_FOUND CPSW_INCLUDE_DIR CPSW_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cpsw DEFAULT_MSG
                                  CPSW_INCLUDE_DIR CPSW_LIBRARY)

if(CPSW_FOUND)
    set(CPSW_LIBRARIES ${CPSW_LIBRARY})
    set(CPSW_INCLUDE_DIRS ${CPSW_INCLUDE_DIR})

    if(NOT TARGET cpsw::cpsw)
        add_library(cpsw::cpsw UNKNOWN IMPORTED)
        set_target_properties(cpsw::cpsw PROPERTIES
            IMPORTED_LOCATION ${CPSW_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${CPSW_INCLUDE_DIR}
        )
    endif()
endif()
