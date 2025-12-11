openmp_dep = dependency('openmp', required: true)

geometry_sources = files(
    'src/SegGeometry.cc',
    'src/SegGeometryCspad2x1V1.cc',
    'src/SegGeometryEpix100V1.cc',
    'src/SegGeometryEpix10kaV1.cc',
    'src/SegGeometryMatrixV1.cc',
    'src/GeometryObject.cc',
    'src/GeometryAccess.cc'
)

geometry_headers = files(
    'GeometryTypes.hh',
    'SegGeometry.hh',
    'SegGeometryCspad2x1V1.hh',
    'SegGeometryEpix100V1.hh',
    'SegGeometryEpix10kaV1.hh',
    'SegGeometryMatrixV1.hh',
    'SegGeometryStore.hh',
    'UtilsCSPAD.hh',
    'GeometryObject.hh',
    'GeometryAccess.hh',
)


geometry_incdir = include_directories(
    '../utils',
    '../calib',
    '../../../xtcdata/xtcdata/xtc'
)

geometry_shlib = shared_library(
    'geometry',
    geometry_sources,
    include_directories: geometry_incdir,
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [utils_shlib, xtc_shlib] ,
    dependencies: [libcurl_dep, openmp_dep]
)

install_headers(
    geometry_headers,
    install_dir: 'include/psalg/geometry'
)
