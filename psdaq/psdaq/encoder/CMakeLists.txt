find_package(usdusb4 REQUIRED)

add_executable(listencoders listencoders.cc)

target_link_libraries(listencoders
    usdusb4::usdusb4
)

install(TARGETS listencoders
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
