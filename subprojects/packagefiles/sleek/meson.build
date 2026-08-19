project('SLEEK',
  ['cpp', 'cuda'],
  version: '1.0.0',
  default_options: [
    'cpp_std=c++17',
    'cuda_std=c++17',
    'b_lto=true',
    'buildtype=release',
  ],
)

cuda = dependency('cuda', required: true)

cuda_arch = [
  '-gencode=arch=compute_86,code=sm_86',
  '-gencode=arch=compute_90,code=sm_90',
]

cpp_args = [
  '-O3',
  '-march=native',
  '-mno-fma',
  '-ffp-contract=off',
  '-Wno-unused-result',
]

cuda_args = [
  '-fmad=false',
  '-Xcompiler=-O3,-march=native,-mno-fma,-ffp-contract=off',
  '-Xcompiler=-Wno-unused-result',
  cuda_arch,
]

sleek_incdir = include_directories('app', '.')

sleek_lib = both_libraries('sleek',
  'app/SingleCompressorLossy.cu',
  include_directories: sleek_incdir,
  cpp_args: cpp_args,
  cuda_args: cuda_args,
  dependencies: cuda,
  install: true,
)

sleek_dep = declare_dependency(
  link_with: sleek_lib,
  include_directories: sleek_incdir,
)

install_headers(
  'app/SingleCompressorLossy.hh',
  subdir: '.',
)

