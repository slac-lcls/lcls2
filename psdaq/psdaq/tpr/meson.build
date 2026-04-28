tpr_incdir = include_directories('../..')

tpr_lib = shared_library('tpr',
  [
    'Module.cc',
    'Client.cc',
  ],
  include_directories: tpr_incdir,
  install_rpath: libdir_install_rpath,
  install: true,
)
