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
    src/Json2Xtc.cc
)

target_include_directories(xtc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${RAPIDJSON_INCLUDE_DIRS}
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
    PulseId.hh
    TransitionId.hh
    Dgram.hh
    Sequence.hh
    TypeId.hh
    VarDef.hh
    Json2Xtc.hh
    DESTINATION include/xtcdata/xtc
)

install(TARGETS xtc
    EXPORT xtcdataTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
