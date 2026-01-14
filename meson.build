
project(
    'lcls',
    'cpp',
    'cython',
    version: '4.3.0',
    default_options: ['cpp_std=c++20']
)
py = import('python').find_installation(pure: false)

libdir_install_rpath = join_paths(get_option('prefix'), get_option('libdir'))
bindir_install_rpath = join_paths(get_option('prefix'), get_option('bindir'))

subdir('xtcdata')
subdir('psalg')
subdir('psana')

if get_option('build_daq')
  subdir('psdaq')
endif
