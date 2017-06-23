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
    xtc/src/Descriptor.cc
    xtc/src/Hdf5Writer.cc
)

target_include_directories(xtc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    #$<INSTALL_INTERFACE:include>
    ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(xtc
    ${HDF5_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
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

#add_executable(descriptor-test
#    app/descriptor-test.cc
#)
#target_link_libraries(descriptor-test
#    xtc
#)
