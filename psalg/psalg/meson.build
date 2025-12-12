cc = meson.get_compiler('cpp')
rt_dep = cc.find_library('rt', required: false)
libcurl_dep = dependency('libcurl')

subdir('utils')
subdir('geometry')
subdir('calib')
subdir('alloc')
subdir('digitizer')
subdir('detector')
subdir('shmem')

psalg_sources = files(
    'calib/src/ArrayIO.cc',
)

psalg_include = include_directories(
    'calib',
    'utils', 
    '../../xtcdata/xtcdata/xtc'
)

psalg_shlib = shared_library(
    'psalg',
    psalg_sources,
    include_directories: psalg_include,
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [utils_shlib, calib_shlib, geometry_shlib, digitizer_shlib, detector_shlib],
    dependencies: [rt_dep]
)

datareader_include = include_directories(
    'calib',
    'utils',
    'detector',
    'geometry',
    '../../xtcdata/xtcdata/xtc'
)

executable(
    'datareader',
    'app/datareader.cc',
    include_directories: datareader_include,
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib, detector_shlib, psalg_shlib, utils_shlib, calib_shlib],
)

hsd_valid_include = include_directories(
    'alloc',
    'digitizer',
    'utils',
    '../../xtcdata/xtcdata/xtc'
)

executable(
    'hsd_valid',
    'tests/hsd_valid.cc',
    include_directories: hsd_valid_include,
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib, psalg_shlib, digitizer_shlib]
)
