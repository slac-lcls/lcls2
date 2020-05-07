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

install(TARGETS epicsArch
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
