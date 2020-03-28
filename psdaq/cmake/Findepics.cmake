find_path(EPICS_PVA_INCLUDE_DIR
          NAMES pva/client.h
          PATH_SUFFIXES include
          HINTS ENV EPICS_BASE
)

find_path(EPICS_OS_INCLUDE_DIR
          NAMES osdMutex.h
          PATH_SUFFIXES include/os/Linux
          HINTS ENV EPICS_BASE
)

find_path(EPICS_COMPILER_INCLUDE_DIR
          NAMES compilerSpecific.h
          PATH_SUFFIXES include/compiler/gcc
          HINTS ENV EPICS_BASE
)

foreach(var IN ITEMS pvAccessCA pvAccess pvData ca Com)
    find_library(EPICS_${var} NAMES ${var})
endforeach(var)

mark_as_advanced(EPICS_FOUND EPICS_PVA_INCLUDE_DIR
                             EPICS_OS_INCLUDE_DIR
                             EPICS_COMPILER_INCLUDE_DIR
                             EPICS_pvAccessCA
                             EPICS_pvAccess
                             EPICS_pvData
                             EPICS_ca
                             EPICS_Com)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(epics DEFAULT_MSG
                                  EPICS_PVA_INCLUDE_DIR
                                  EPICS_OS_INCLUDE_DIR
                                  EPICS_COMPILER_INCLUDE_DIR
                                  EPICS_pvAccessCA
                                  EPICS_pvAccess
                                  EPICS_pvData
                                  EPICS_ca
                                  EPICS_Com
)

if(EPICS_FOUND)
    set(EPICS_LIBRARIES "${EPICS_pvAccessCA}" "${EPICS_pvAccess}" "${EPICS_pvData}" "${EPICS_ca}" "${EPICS_Com}")
    set(EPICS_INCLUDE_DIRS "${EPICS_PVA_INCLUDE_DIR}"
                           "${EPICS_OS_INCLUDE_DIR}"
                           "${EPICS_COMPILER_INCLUDE_DIR}")

    if(NOT TARGET epics::pvAccessCA)
        add_library(epics::pvAccessCA UNKNOWN IMPORTED)
        set_target_properties(epics::pvAccessCA PROPERTIES
            IMPORTED_LOCATION ${EPICS_pvAccessCA}
            INTERFACE_INCLUDE_DIRECTORIES "${EPICS_INCLUDE_DIRS}"
        )
    endif()

    if(NOT TARGET epics::pvAccess)
        add_library(epics::pvAccess UNKNOWN IMPORTED)
        set_target_properties(epics::pvAccess PROPERTIES
            IMPORTED_LOCATION ${EPICS_pvAccess}
            INTERFACE_INCLUDE_DIRECTORIES "${EPICS_INCLUDE_DIRS}"
        )
    endif()

    if(NOT TARGET epics::pvData)
        add_library(epics::pvData UNKNOWN IMPORTED)
        set_target_properties(epics::pvData PROPERTIES
            IMPORTED_LOCATION ${EPICS_pvData}
            INTERFACE_INCLUDE_DIRECTORIES "${EPICS_INCLUDE_DIRS}"
        )
    endif()

    if(NOT TARGET epics::ca)
        add_library(epics::ca UNKNOWN IMPORTED)
        set_target_properties(epics::ca PROPERTIES
            IMPORTED_LOCATION ${EPICS_ca}
            INTERFACE_INCLUDE_DIRECTORIES "${EPICS_INCLUDE_DIRS}"
        )
    endif()

    if(NOT TARGET epics::Com)
        add_library(epics::Com UNKNOWN IMPORTED)
        set_target_properties(epics::Com PROPERTIES
            IMPORTED_LOCATION ${EPICS_Com}
            INTERFACE_INCLUDE_DIRECTORIES "${EPICS_INCLUDE_DIRS}"
        )
    endif()
endif()
