project('PFPL',
  ['cpp', 'cuda'],
  version: '0.0.0',
  default_options: [
    'cpp_std=c++17',
    'cuda_std=c++17',
    'b_lto=true',
    'buildtype=release',
    'warning_level=1',
    'b_ndebug=if-release',
    'cpp_args=-march=native -mno-fma -ffp-contract=off -Wno-unused-result',
  ]
)

cuda_compiler = meson.get_compiler('cuda')

conda_base = get_option('conda_prefix')
cuda_dep = declare_dependency(
    include_directories: [
      include_directories(join_paths(conda_base, 'targets/x86_64-linux/include/')),
      include_directories(join_paths(conda_base, 'targets/x86_64-linux/include/cccl/')),
    ],
    link_args: [
      '-L' + join_paths(join_paths(conda_base, 'targets/x86_64-linux/lib/')),
      '-lcudart',
      '-lcuda',
   ]
)

add_project_arguments('-arch=sm_80', '-arch=sm_86', '-arch=sm_90', '-arch=sm_100', language: 'cuda')
add_project_arguments('-fmad=false', '-Xcompiler=-march=native,-mno-fma,-ffp-contract=off,-Wno-unused-result', language: 'cuda')

pfpl_inc_dir = include_directories('src', 'app')

pfpl_lib = both_libraries('pfpl',
  'app/f32_abs_comp_gpu.cu',
  include_directories: pfpl_inc_dir,
  dependencies: cuda_dep,
  install: true,
  cuda_args: ['-dc'],
  pic: true
)

install_headers('app/f32_abs_comp_gpu.hh')

pfpl_dep = declare_dependency(
  link_with: pfpl_lib,
  include_directories: pfpl_inc_dir,
)
