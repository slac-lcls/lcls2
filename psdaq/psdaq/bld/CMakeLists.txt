
add_library(bld SHARED
    HpsEventIterator.cc
    Server.cc
    Client.cc
)

target_include_directories(bld PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS bld
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
