project('LC',
  ['cpp', 'cuda'],
  version: '0.0.0',
  default_options: [
    'buildtype=release',
    'cpp_std=c++17',
    'cuda_std=c++17',
    'b_staticpic=true',
  ])

# Compilers
cpp = meson.get_compiler('cpp')
cuda = meson.get_compiler('cuda')

cpp_args = [
  '-O3',
  '-march=native',
  '-mno-fma',
  '-ffp-contract=off',
  '-Wno-unused-result'
]

cuda_args = [
  '-O3',
  '-fmad=false',
  '-Xcompiler=-Wno-unused-result',
  '-Xcompiler=-O3,-march=native,-mno-fma,-ffp-contract=off',
  '-gencode=arch=compute_80,code=sm_80',
  '-gencode=arch=compute_86,code=sm_86',
  '-gencode=arch=compute_90,code=sm_90',
  '-gencode=arch=compute_100,code=sm_100',
]

lc_inc_dir = include_directories('.')
lc_app_dir  = include_directories('app')

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

lc_static = static_library('lc',
  'app/lc-compressor-QUANT_ABS_0_f32-BIT_4-RZE_1.cu',
  include_directories: [lc_inc_dir, lc_app_dir],
  cpp_args: cpp_args,
  cuda_args: cuda_args,
  dependencies: cuda_dep,
  install: true,
)

lc_shared = shared_library('lc',
  'app/lc-compressor-QUANT_ABS_0_f32-BIT_4-RZE_1.cu',
  include_directories: [lc_inc_dir, lc_app_dir],
  cpp_args: cpp_args,
  cuda_args: cuda_args,
  dependencies: cuda_dep,
  install: true,
)

lc_dep = declare_dependency(
  link_with: lc_shared,
  include_directories: [
    lc_inc_dir,
    lc_app_dir,
  ]
)

install_headers(
  'app/lc-compressor-QUANT_ABS_0_f32-BIT_4-RZE_1.hh',
  subdir: '.',
)
