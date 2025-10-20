message(STATUS "xtc/CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "xtc/CMAKE_SOURCE_DIR         = ${CMAKE_SOURCE_DIR}")

add_library(xtc SHARED
    src/TransitionId.cc
    src/XtcIterator.cc
    src/TimeStamp.cc
    src/Level.cc
    src/TypeId.cc
    src/XtcFileIterator.cc
    src/ShapesData.cc
    src/NamesIter.cc
    src/ConfigIter.cc
    src/DataIter.cc
    src/Smd.cc
    src/XtcUpdateIter.cc
)

target_include_directories(xtc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>   # one level up
    $<INSTALL_INTERFACE:include>
)

# A static version of the xtc library is needed for kcuStatus
add_library(staticXtc STATIC
    src/TransitionId.cc
    src/XtcIterator.cc
    src/TimeStamp.cc
    src/Level.cc
    src/TypeId.cc
    src/XtcFileIterator.cc
    src/ShapesData.cc
    src/NamesIter.cc
    src/ConfigIter.cc
    src/DataIter.cc
    src/Smd.cc
    src/XtcUpdateIter.cc
)

target_include_directories(staticXtc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>   # one level up
    $<INSTALL_INTERFACE:include>
)


install(FILES
    Level.hh
    NamesId.hh
    NamesLookup.hh
    NameIndex.hh
    ShapesData.hh
    BlockDgram.hh
    Array.hh
    XtcFileIterator.hh
    Damage.hh
    NamesIter.hh
    ConfigIter.hh
    DataIter.hh
    Src.hh
    Xtc.hh
    DescData.hh
    TimeStamp.hh
    XtcIterator.hh
    TransitionId.hh
    Dgram.hh
    TypeId.hh
    VarDef.hh
    Smd.hh
    XtcUpdateIter.hh
    DESTINATION include/xtcdata/xtc
)

install(TARGETS xtc staticXtc
    EXPORT xtcdataTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include/xtcdata/xtc
        FILES_MATCHING PATTERN "*.hh")

