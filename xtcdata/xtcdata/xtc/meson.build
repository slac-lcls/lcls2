xtc_sources = files(
    'src/ConfigIter.cc',
    'src/DataIter.cc',
    'src/Level.cc',
    'src/NamesIter.cc',
    'src/ShapesData.cc',
    'src/Smd.cc',
    'src/TimeStamp.cc',
    'src/TransitionId.cc',
    'src/TypeId.cc',
    'src/XtcFileIterator.cc',
    'src/XtcIterator.cc',
    'src/XtcUpdateIter.cc'
)

xtc_headers = files(
    'Level.hh',
    'NamesId.hh',
    'NamesLookup.hh',
    'NameIndex.hh',
    'ShapesData.hh',
    'BlockDgram.hh',
    'Array.hh',
    'XtcFileIterator.hh',
    'Damage.hh',
    'NamesIter.hh',
    'ConfigIter.hh',
    'DataIter.hh',
    'Src.hh',
    'Xtc.hh',
    'DescData.hh',
    'TimeStamp.hh',
    'XtcIterator.hh',
    'TransitionId.hh',
    'Dgram.hh',
    'TypeId.hh',
    'VarDef.hh',
    'Smd.hh',
    'XtcUpdateIter.hh'
)

xtc_shlib = shared_library(
    'xtc',
    xtc_sources,
    install: true,
    install_rpath: libdir_install_rpath
)

install_headers(
    xtc_headers,
    install_dir: 'include/xtcdata/xtc',
)
