
project(
    'lcls',
    ['cpp', 'cuda'],
    'cython',
    version: '4.3.0',
    default_options: ['cpp_std=c++20']
)

conda_base = get_option('conda_prefix')
py = import('python').find_installation(pure: false)
python_dep = py.dependency(embed: true)

# Have to set the linker flags manually, as nvcc's linker conflicts with
# the gcc type linker args that conda sets.
add_project_link_arguments(
  [
    '-Wl,-O2',
    '-Wl,--sort-common',
    '-Wl,--as-needed',
    '-Wl,--gc-sections',
    '-Wl,-z,relro',
    '-Wl,-z,now',
    '-Wl,--disable-new-dtags',
    '-Wl,--allow-shlib-undefined',
    '-Wl,-rpath,' + join_paths(conda_base, 'lib'),
    '-Wl,-rpath-link,' + join_paths(conda_base, 'lib'),
    '-L' + join_paths(conda_base, 'lib'),
    '-L' + join_paths(conda_base, 'targets/x86_64-linux/lib'),
    '-L' + join_paths(conda_base, 'targets/x86_64-linux/lib/stubs'),
  ],
  language: 'cpp'
)

add_project_link_arguments(
  [
    '-Xlinker=-O2',
    '-Xlinker=--as-needed',
    '-Xlinker=--gc-sections',
    '-Xlinker=-z,relro',
    '-Xlinker=-z,now',
    '-Xlinker=-rpath,' + join_paths(conda_base, 'lib'),
    '-L' + join_paths(conda_base, 'lib'),
    '-L,' + join_paths(conda_base, 'targets/x86_64-linux/lib'),
  ],
  language: 'cuda'
)

libdir_install_rpath = join_paths(get_option('prefix'), get_option('libdir'))
bindir_install_rpath = join_paths(get_option('prefix'), get_option('bindir'))

subdir('xtcdata')
subdir('psalg')
subdir('psana')

if get_option('build_daq')
  subdir('psdaq')
endif
