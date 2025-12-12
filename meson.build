
project(
    'lcls',
    'cpp',
    version: '4.3.0',
    default_options: ['cpp_std=c++20']
)
py = import('python').find_installation(pure: false)

libdir_install_rpath = join_paths(get_option('prefix'), get_option('libdir'))

subdir('xtcdata/xtcdata')
subdir('psalg/psalg')
