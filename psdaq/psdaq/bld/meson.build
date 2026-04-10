
bld_inc_dir = include_directories('../..')

bld_lib = shared_library('bld',
  [
    'HpsEventIterator.cc',
    'Server.cc',
    'Client.cc',
  ],
  include_directories: [bld_inc_dir],
  dependencies: [
    threads_dep,
  ],
  install_rpath: libdir_install_rpath,
  install: true,
)
