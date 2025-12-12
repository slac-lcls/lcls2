digitizer_sources = files(
    'src/Hsd.cc',
    'src/Stream.cc',
)

digitizer_headers = files(
    'Hsd.hh',
    'Stream.hh'
)
 
digitizer_shlib = shared_library(
    'digitizer',
    digitizer_sources,
    include_directories: [psalg_incdir, xtcdata_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib, utils_shlib]
)

install_headers(
    digitizer_headers,
    install_dir: 'include/psalg/digitizer'
)
