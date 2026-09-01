project(
    'lcls',
    'cpp',
    'cython',
    version: '4.3.0',
    default_options: ['cpp_std=c++20', 'libdir=lib']
)

conda_base = get_option('conda_prefix')
cuda_base = get_option('custom_cuda_path')
cc = meson.get_compiler('cpp')
py = import('python').find_installation(pure: false)
python_dep = py.dependency(embed: true)

numpy_inc = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
numpy_dep = declare_dependency(include_directories: include_directories(numpy_inc))

# Have to set the linker flags manually, as nvcc's linker conflicts with
# the gcc linker args that conda sets.
# Only applies when a Conda environment is actually in use (conda_prefix set).
# In a manylinux container build, conda_base is empty and these flags are
# skipped entirely so we fall back to the compiler's default system paths.
if conda_base != ''
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
    '-L' + join_paths(conda_base, 'targets/x86_64-linux/lib'),
  ],
  language: 'cuda'
)
endif

# EPICS is only relevant to DAQ libraries. Without this guard, every shared
# library in the project (including xtcdata/psalg/psana, which never touch
# EPICS) gets an EPICS path baked into its RPATH even when build_daq=false.
epics_lib = join_paths(get_option('epics_base'), 'lib', get_option('epics_host_arch'))
if get_option('build_daq')
  libdir_install_rpath = join_paths(get_option('prefix'), get_option('libdir')) + ':' + epics_lib
else
  libdir_install_rpath = join_paths(get_option('prefix'), get_option('libdir'))
endif
bindir_install_rpath = join_paths(get_option('prefix'), get_option('bindir'))

subdir('xtcdata')
subdir('psalg')
subdir('psana')

cuda_compiler = find_program('nvcc', required: false)

if cuda_compiler.found()
  add_languages('cuda', native: false) # Python 3.9
  cmake = import('cmake')
  cusz_proj = cmake.subproject('cusz')
  cuszp_proj = cmake.subproject('cuszp')
endif

if get_option('build_daq')
  subdir('psdaq')
endif
