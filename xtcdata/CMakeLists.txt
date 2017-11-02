
add_library(xtc SHARED
    xtc/src/BldInfo.cc
    xtc/src/L1AcceptEnv.cc
    xtc/src/Sequence.cc
    xtc/src/TransitionId.cc
    xtc/src/XtcIterator.cc
    xtc/src/ClockTime.cc
    xtc/src/Level.cc
    xtc/src/Src.cc
    xtc/src/TypeId.cc
    xtc/src/DetInfo.cc
    xtc/src/ProcInfo.cc
    xtc/src/TimeStamp.cc
    xtc/src/XtcFileIterator.cc
    xtc/src/ShapesData.cc
    xtc/src/NamesIter.cc
)

target_include_directories(xtc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

add_executable(xtcreader
    app/xtcreader.cc
)
target_link_libraries(xtcreader
    xtc
)

add_executable(xtcwriter
    app/xtcwriter.cc
)
target_link_libraries(xtcwriter
    xtc
)
