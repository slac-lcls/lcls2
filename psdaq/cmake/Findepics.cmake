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

foreach(var IN ITEMS pvAccess pvData)
    find_library(EPICS_${var} NAMES ${var})
endforeach(var)

mark_as_advanced(EPICS_FOUND EPICS_PVA_INCLUDE_DIR 
                             EPICS_OS_INCLUDE_DIR
                             EPICS_COMPILER_INCLUDE_DIR 
                             EPICS_pvAccess
                             EPICS_pvData)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(epics DEFAULT_MSG
                                  EPICS_PVA_INCLUDE_DIR 
                                  EPICS_OS_INCLUDE_DIR
                                  EPICS_COMPILER_INCLUDE_DIR 
                                  EPICS_pvAccess
                                  EPICS_pvData
)

if(EPICS_FOUND)
    set(EPICS_LIBRARIES "${EPICS_pvAccess}" "${EPICS_pvData}")
    set(EPICS_INCLUDE_DIRS "${EPICS_PVA_INCLUDE_DIR}" 
                           "${EPICS_OS_INCLUDE_DIR}" 
                           "${EPICS_COMPILER_INCLUDE_DIR}")

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
endif()
