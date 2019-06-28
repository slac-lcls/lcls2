
add_library(epicstools SHARED
    EpicsPVA.cc
    PvServer.cc
)

target_include_directories(epicstools PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(epicstools
    epics::pvAccess
    epics::pvData
    #epics::pvaClient
    #epics::ca
)

install(TARGETS epicstools
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
