add_library(alloc INTERFACE)

target_link_libraries(alloc INTERFACE
    xtcdata::xtc
)

target_include_directories(alloc INTERFACE
    $<INSTALL_INTERFACE:include>
)

install(TARGETS alloc
    EXPORT psalgTargets
    INCLUDES DESTINATION include/psalg/alloc
)

install(FILES
        AllocArray.hh
        Allocator.hh
        DESTINATION include/psalg/alloc)