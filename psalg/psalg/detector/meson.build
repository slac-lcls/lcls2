detector_sources = files(
    'src/DetectorTypes.cc',
    'src/Detector.cc',
    'src/DetectorStore.cc',
    'src/UtilsConfig.cc',
    'src/AreaDetector.cc',
    'src/AreaDetectorJungfrau.cc',
    'src/AreaDetectorPnccd.cc',
    'src/AreaDetectorOpal.cc',
    'src/AreaDetectorEpix100a.cc',
    'src/AreaDetectorStore.cc'
)

detector_headers = files(
    'DetectorTypes.hh',
    'Detector.hh',
    'DetectorStore.hh',
    'UtilsConfig.hh',
    'AreaDetector.hh',
    'AreaDetectorStore.hh',
    'AreaDetectorJungfrau.hh',
    'AreaDetectorPnccd.hh',
    'AreaDetectorOpal.hh',
    'AreaDetectorEpix100a.hh',
    'DataSourceSimulator.hh',
)

detector_shlib = shared_library(
    'detector',
    detector_sources,
    include_directories: [psalg_incdir, xtcdata_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib, utils_shlib, calib_shlib, geometry_shlib]
)

install_headers(
    detector_headers,
    install_dir: 'include/psalg/detector'
)
