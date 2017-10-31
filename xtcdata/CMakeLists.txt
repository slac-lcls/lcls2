
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
)

target_include_directories(xtc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

#add_library(dgram SHARED
#    app/dgram.cc
#)
#
#target_link_libraries(dgram
#    xtc
#    ${PYTHON_LIBRARIES}
#)
#
#target_include_directories(dgram PUBLIC
#    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> 
#    ${PYTHON_INCLUDE_DIRS}
#    ${NUMPY_INCLUDE_DIRS}
#)
#
#set_target_properties(dgram PROPERTIES PREFIX "")

target_link_libraries(xtc
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
