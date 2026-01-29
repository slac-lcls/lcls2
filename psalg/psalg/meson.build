cc = meson.get_compiler('cpp')
rt_dep = cc.find_library('rt', required: false)
libcurl_dep = dependency('libcurl')
threads_dep = dependency('threads')

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

psalg_shlib = shared_library(
    'psalg',
    psalg_sources,
    include_directories: [psalg_incdir, xtcdata_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [utils_shlib, calib_shlib, geometry_shlib, digitizer_shlib, detector_shlib],
    dependencies: [rt_dep]
)

executable(
    'datareader',
    'app/datareader.cc',
    include_directories: [psalg_incdir, xtcdata_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib, detector_shlib, psalg_shlib, utils_shlib, calib_shlib],
)

executable(
    'hsd_valid',
    'tests/hsd_valid.cc',
    include_directories: [psalg_incdir, xtcdata_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib, psalg_shlib, digitizer_shlib]
)
