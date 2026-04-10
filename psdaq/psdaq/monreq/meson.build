monreq_inc_dir = include_directories('../..')

monReqServer = executable('monReqServer',
  'monReqServer.cc',
  include_directories: [
    monreq_inc_dir,
    include_directories('../../../psalg')
  ],
  dependencies: [
    psalg_utils_dep,
    xtcdata_dep,
    zmq_dep,
    prometheus_pull_dep,
    threads_dep
  ],
  link_with: [
    eventBuilder_lib,
    service_lib,
    shmemserver_shlib,
    collection_lib,
    utilities_lib,
    exporter_lib
  ],
  install_rpath: libdir_install_rpath,
  install: true
)
