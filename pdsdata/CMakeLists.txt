
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

add_library(dgram SHARED
    app/pydgconvert.cc
)

target_link_libraries(dgram
    xtc
    ${PYTHON_LIBRARIES}
)

target_include_directories(dgram PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> 
    ${PYTHON_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
)

set_target_properties(dgram PROPERTIES PREFIX "")

add_library(testType SHARED
    app/testProg.cc
)

target_link_libraries(testType
    ${PYTHON_LIBRARIES}
)

target_include_directories(testType PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> 
    ${PYTHON_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIRS}
)

set_target_properties(testType PROPERTIES PREFIX "")

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
