add_library(hexanode SHARED
    src/hexanode.cc
)

target_link_libraries(hexanode
    roentdek::roentdek
)

target_include_directories(hexanode PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
