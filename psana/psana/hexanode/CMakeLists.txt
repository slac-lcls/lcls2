add_library(hexanode SHARED
    #src/hexanode.cc
    src/cfib.cc
    src/wrap_resort64c.cc
    src/LMF_IO.cc
    src/SortUtils.cc
)

target_link_libraries(hexanode
    roentdek::resort64c
)

target_include_directories(hexanode PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
