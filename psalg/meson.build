psalg_incdir = include_directories(
    '.'
)
subdir('psalg')

install_subdir(
  'psalg',
  install_dir: py.get_install_dir(),
  exclude_directories: [
    'alloc/',
    'app/',
    'calib/',
    'detector/',
    'digitizer/',
    'geometry/',
    'shmem/',
    'tests/',
    'utils/src/',
  ],
  exclude_files: [
    'CMakeLists.txt',
    'meson.build',
    'utils/CMakeLists.txt',
    'utils/ctest_utils.hh',
    'utils/DirFileIterator.hh',
    'utils/Logger.hh',
    'utils/MacTimeFix.hh',
    'utils/meson.build',
    'utils/NDArrayGenerators.hh',
    'utils/SysLog.hh',
    'utils/trim.hh',
    'utils/Utils.hh',
  ],
)
