add_library(xtcdata SHARED
    xtc/src/BldInfo.cc
    xtc/src/Sequence.cc
    xtc/src/TransitionId.cc
    xtc/src/XtcIterator.cc
    xtc/src/TimeStamp.cc
    xtc/src/Level.cc
    xtc/src/Src.cc
    xtc/src/TypeId.cc
    xtc/src/DetInfo.cc
    xtc/src/ProcInfo.cc
    xtc/src/PulseId.cc
    xtc/src/XtcFileIterator.cc
    xtc/src/ShapesData.cc
    xtc/src/NamesIter.cc
)

target_include_directories(xtcdata PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

add_executable(xtcreader
    app/xtcreader.cc
)
target_link_libraries(xtcreader
    xtcdata
)

add_executable(xtcwriter
    app/xtcwriter.cc
)
target_link_libraries(xtcwriter
    xtcdata
)

add_executable(smdwriter
    app/smdwriter.cc
)
target_link_libraries(smdwriter
    xtcdata
)

install(FILES
    xtc/BldInfo.hh
    xtc/Level.hh
    xtc/ShapesData.hh
    xtc/XtcFileIterator.hh
    xtc/Damage.hh
    xtc/NamesIter.hh
    xtc/Src.hh
    xtc/Xtc.hh
    xtc/DescData.hh
    xtc/ProcInfo.hh
    xtc/TimeStamp.hh
    xtc/XtcIterator.hh
    xtc/DetInfo.hh
    xtc/PulseId.hh
    xtc/TransitionId.hh
    xtc/Dgram.hh
    xtc/Sequence.hh
    xtc/TypeId.hh
    xtc/VarDef.hh
    DESTINATION include/xtcdata/xtc
)

install(TARGETS xtcdata
                xtcreader
                xtcwriter
                smdwriter
    EXPORT xtcdataTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(EXPORT xtcdataTargets
    FILE xtcdataConfig.cmake
    DESTINATION lib/cmake/xtcdata
)
