project(
  'lcls',
  'cpp',
  'cython',
  version: '4.3.0',
  default_options: [
    'cpp_std=c++20',
    'libdir=lib',
    'b_lto=true',
    'b_ndebug=true',
  ]
)

conda_base = get_option('conda_prefix')
cuda_base = get_option('custom_cuda_path')
cc = meson.get_compiler('cpp')
ar = find_program('gcc-ar', required: true)
py = import('python').find_installation(pure: false)
python_dep = py.dependency(embed: true)

# Python 3.11
#numpy_dep = dependency('numpy')

# Python 3.9
numpy_inc = run_command('python3', ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
numpy_dep = declare_dependency(include_directories: include_directories(numpy_inc))

# Have to set the linker flags manually, as nvcc's linker conflicts with
# the gcc linker args that conda sets.
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

epics_lib = join_paths(get_option('epics_base'), 'lib', get_option('epics_host_arch'))
libdir_install_rpath = join_paths(get_option('prefix'), get_option('libdir')) + ':' + epics_lib
bindir_install_rpath = join_paths(get_option('prefix'), get_option('bindir'))

subdir('xtcdata')
subdir('psalg')
subdir('psana')

cuda_compiler = find_program('nvcc', required: false)

if cuda_compiler.found()
  add_languages('cuda', native: false) # Python 3.9
  cmake = import('cmake')
  cmk_opts = cmake.subproject_options()
  cmk_opts.add_cmake_defines({'CMAKE_BUILD_TYPE': 'RelWithDebInfo'})
  cmk_opts.add_cmake_defines({'CMAKE_CUDA_ARCHITECTURES': '86;90'})
  cmk_opts.set_override_option('cpp_std', 'c++17')
  cmk_opts.set_override_option('cuda_std', 'c++17')
  # Build fails with the following (which the old method had)?
  #cmk_opts.append_compile_args('cuda', '-rdc=true')
  cmk_opts.set_override_option('warning_level', '0')
  cmk_opts.append_compile_args('cpp', '-w')
  cmk_opts.append_compile_args('cuda', '-Xcompiler=-w')

  # 6/16/26: Comment these out as they produce lots of warnings
  #cusz_proj = cmake.subproject('cusz', options: cmk_opts)
  #cuszp_proj = cmake.subproject('cuszp', options: cmk_opts)
endif

if get_option('build_daq')
  subdir('psdaq')
endif
