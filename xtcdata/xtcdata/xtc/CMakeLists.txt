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
)

target_include_directories(xtc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Why is this a separate library?  Because of https://bugzilla.redhat.com/show_bug.cgi?id=1519073
add_library(j2x SHARED
    src/Json2Xtc.cc
)

target_include_directories(j2x PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    $ENV{CONDA_PREFIX}/include
)

target_link_libraries(j2x PUBLIC xtc)

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

install(TARGETS xtc j2x
    EXPORT xtcdataTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
