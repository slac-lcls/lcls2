project(
  'sleek',
  ['cpp', 'cuda'],
  version: '1.0.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=1',
    'buildtype=release',
  ],
)

cuda = dependency('cuda', required: true)

cpp_args = [
  '-O3',
  '-march=native',
  '-mno-fma',
  '-ffp-contract=off',
  '-Wno-unused-result',
]

cuda_args = [
  '-fmad=false',
  #'-Wno-unused-result',
]

cuda_cpp_args = [
  '-Xcompiler=-O3,-march=native,-mno-fma,-ffp-contract=off,-Wno-unused-result'
]

sleek_incdir = include_directories('app', '.')

sleek_lib = both_libraries('sleek',
  'app/SingleCompressorLossy.cu',
  include_directories: sleek_incdir,
  cpp_args: cpp_args,
  cuda_args: cuda_args + cuda_cpp_args,
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

