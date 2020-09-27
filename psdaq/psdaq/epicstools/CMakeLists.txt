
add_library(epicstools SHARED
    EpicsProviders.cc
    EpicsPVA.cc
    MonTracker.cc
    PvServer.cc
)

target_include_directories(epicstools PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(epicstools
    epics::pvAccessCA
    epics::pvAccess
    epics::pvData
    epics::ca
    epics::Com
)

install(TARGETS epicstools
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
