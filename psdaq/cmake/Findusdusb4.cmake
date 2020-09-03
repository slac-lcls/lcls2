find_path(USDUSB4_INCLUDE_DIR
          NAMES libusdusb4.h
)

foreach(var IN ITEMS usdusb4 udev usb-1.0)
    find_library(USDUSB4_${var} ${var})
endforeach(var)

mark_as_advanced(USDUSB4_FOUND USDUSB4_INCLUDE_DIR
                               USDUSB4_usdusb4
                               USDUSB4_udev
                               USDUSB4_usb-1.0)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(usdusb4 DEFAULT_MSG
                                  USDUSB4_INCLUDE_DIR
                                  USDUSB4_usdusb4
                                  USDUSB4_udev
                                  USDUSB4_usb-1.0)

if(USDUSB4_FOUND)
    message(STATUS "usdusb4 found.")
    set(USDUSB4_LIBRARIES ${USDUSB4_usdusb4} ${USDUSB4_udev} ${USDUSB4_usb-1.0})
    set(USDUSB4_INCLUDE_DIRS ${USDUSB4_INCLUDE_DIR})
    message(DEBUG "usdusb4 libraries: ${USDUSB4_LIBRARIES}")
    message(DEBUG "usdusb4 headers: ${USDUSB4_INCLUDE_DIRS}")

    if(NOT TARGET usdusb4::libusdusb4)
        add_library(usdusb4::libusdusb4 UNKNOWN IMPORTED)
        set_target_properties(usdusb4::libusdusb4 PROPERTIES
            IMPORTED_LOCATION ${USDUSB4_usdusb4}
            INTERFACE_INCLUDE_DIRECTORIES "${USDUSB4_INCLUDE_DIRS}")
    endif()

    if(NOT TARGET usdusb4::udev)
        add_library(usdusb4::udev UNKNOWN IMPORTED)
        set_target_properties(usdusb4::udev PROPERTIES
            IMPORTED_LOCATION ${USDUSB4_udev}
            INTERFACE_INCLUDE_DIRECTORIES "${USDUSB4_INCLUDE_DIRS}")
    endif()

    if(NOT TARGET usdusb4::usb-1.0)
        add_library(usdusb4::usb-1.0 UNKNOWN IMPORTED)
        set_target_properties(usdusb4::usb-1.0 PROPERTIES
            IMPORTED_LOCATION ${USDUSB4_usb-1.0}
            INTERFACE_INCLUDE_DIRECTORIES "${USDUSB4_INCLUDE_DIRS}")
    endif()

    if(NOT TARGET usdusb4::usdusb4)
        add_library(usdusb4::usdusb4 INTERFACE IMPORTED)
        set_target_properties(usdusb4::usdusb4 PROPERTIES
            INTERFACE_LINK_LIBRARIES "${USDUSB4_LIBRARIES}")
    endif()
endif()
