calib_sources = files(
    'src/ArrayIO.cc',
    'src/CalibParsTypes.cc',
    'src/AreaDetectorTypes.cc',
    'src/CalibPars.cc',
    'src/CalibParsDB.cc',
    'src/CalibParsDBTypes.cc',
    'src/CalibParsDBWeb.cc',
    'src/CalibParsDBStore.cc',
    'src/CalibParsEpix100a.cc',
    'src/CalibParsStore.cc',
    'src/Query.cc',
    'src/MDBWebUtils.cc'
)

calib_headers = files(
    'Types.hh',
    'AreaDetectorTypes.hh',
    'NDArray.hh',
    'ArrayIO.hh',
    'CalibParsTypes.hh',
    'CalibParsDBTypes.hh',
    'CalibPars.hh',
    'CalibParsDB.hh',
    'CalibParsDBWeb.hh',
    'CalibParsDBStore.hh',
    'CalibParsEpix100a.hh',
    'CalibParsStore.hh',
    'Query.hh',
    'MDBWebUtils.hh'
)

calib_shlib = shared_library(
    'calib',
    calib_sources,
    include_directories: [psalg_incdir, xtcdata_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [utils_shlib, geometry_shlib, xtc_shlib],
    dependencies: [libcurl_dep],
)

install_headers(
    calib_headers,
    install_dir: 'include/psalg/calib'
)
