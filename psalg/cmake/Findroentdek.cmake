# find the include path and store it in ROENTDEK_INCLUDE_DIR
find_path(ROENTDEK_INCLUDE_DIR
    NAMES roentdek/resort64c.h
)

# find the library and store it in ROENTDEK_LIBRARY
find_library(ROENTDEK_LIBRARY NAMES Resort64c_x64
)

# if running cmake in GUI mode, hide these variables unless "advanced"
# is selected.  does nothing in script-mode, which lcls2 uses.
mark_as_advanced(ROENTDEK_FOUND ROENTDEK_INCLUDE_DIR ROENTDEK_LIBRARY)

include(FindPackageHandleStandardArgs)

# check to see if all the right variables are found, and if so
# set ROENTDEK_FOUND.  cmake variables are case-sensitive, but
# function calls are not, but this function uppercases the variable
# "roentdek"
find_package_handle_standard_args(roentdek DEFAULT_MSG
                                  ROENTDEK_INCLUDE_DIR ROENTDEK_LIBRARY)

if(ROENTDEK_FOUND)
    set(ROENTDEK_LIBRARIES ${ROENTDEK_LIBRARY})
    set(ROENTDEK_INCLUDE_DIRS ${ROENTDEK_INCLUDE_DIR})

    # I think this sets up the machinery needed to locate
    # headers/libraries using the "roentdek::resort64c" syntax
    # in CMakeLists.txt
    if(NOT TARGET roentdek::resort64c)
        add_library(roentdek::resort64c UNKNOWN IMPORTED)
        set_target_properties(roentdek::resort64c PROPERTIES
            IMPORTED_LOCATION ${ROENTDEK_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${ROENTDEK_INCLUDE_DIR}
        )
    endif()
endif()
