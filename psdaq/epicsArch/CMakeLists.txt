find_package(PythonLibs REQUIRED)

add_executable(epicsArch
    EpicsArchMonitor.cc
    EpicsMonitorPv.cc
    PvConfigFile.cc
    epicsArch.cc
)

target_link_libraries(epicsArch
    drpbase
    trigger
    epicstools
    readline
)

add_executable(epicsArchVerify
    PvConfigFile.cc
    epicsArchVerify.cc
)

target_include_directories(epicsArchVerify PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(epicsArchVerify
    psalg::utils
)

install(TARGETS epicsArch epicsArchVerify
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
