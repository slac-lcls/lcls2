project('PFPL',
  ['cpp', 'cuda'],
  version: '0.0.0',
  default_options: [
    'cpp_std=c++17',
    'cuda_std=c++17',
    'b_lto=true',
    'buildtype=release',
    'b_ndebug=if-release',
  ]
)

cuda = dependency('cuda', required: true)

cpp_compiler = meson.get_compiler('cpp')
cuda_compiler = meson.get_compiler('cuda')

cuda_arch = [
  '-gencode=arch=compute_86,code=sm_86',
  '-gencode=arch=compute_90,code=sm_90',
]

cpp_args = [
  '-O3',
  '-march=native',
  '-mno-fma',
  '-ffp-contract=off',
  '-Wno-unused-result'
]

cuda_args = [
  '-fmad=false',
  '-Xcompiler=-O3,-march=native,-mno-fma,-ffp-contract=off',
  '-Xcompiler=-Wno-unused-result',
  cuda_arch,
]

pfpl_inc_dir = include_directories('src', 'app')

conda_base = get_option('conda_prefix')
cuda_dep = declare_dependency(
    include_directories: [
      include_directories(join_paths(conda_base, 'targets/x86_64-linux/include/')),
      include_directories(join_paths(conda_base, 'targets/x86_64-linux/include/cccl/')),
    ],
    link_args: [
      cuda_arch,
      '-L' + join_paths(join_paths(conda_base, 'targets/x86_64-linux/lib/')),
      '-lcudart',
      '-lcuda',
   ]
)

pfpl_lib = both_libraries('pfpl',
  'app/f32_abs_comp_gpu.cu',
  include_directories: pfpl_inc_dir,
  cpp_args: cpp_args,
  cuda_args: [cuda_args, '-dc'],
  dependencies: cuda_dep,
  install: true,
  pic: true,
)

install_headers('app/f32_abs_comp_gpu.hh')

pfpl_dep = declare_dependency(
  link_with: pfpl_lib,
  include_directories: pfpl_inc_dir,
)
