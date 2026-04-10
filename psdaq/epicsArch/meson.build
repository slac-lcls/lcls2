epics_arch_inc_dir = include_directories('../..')
psalg_inc_dir = include_directories('../../psalg')

executable('epicsArch',
  [
    '../drp/TebReceiver.cc',
    'EpicsArchMonitor.cc',
    'EpicsMonitorPv.cc',
    'PvConfigFile.cc',
    'epicsArch.cc',
  ],
  include_directories: [
    epics_arch_inc_dir,
    psalg_inc_dir
  ],
  dependencies: [
    xtcdata_dep,
    drp_dep,
    python_dep,
    epics_dep,
    zmq_dep,
    threads_dep,
    prometheus_pull_dep,
  ],
  install: true,
  install_rpath: libdir_install_rpath,
  link_with: [
    epicstools_lib,
    collection_lib,
    service_lib,
    exporter_lib,
    contributor_lib,
    utilities_lib,
  ],
)

executable('epicsArchVerify',
  [
    'PvConfigFile.cc',
    'epicsArchVerify.cc',
  ],
  include_directories: [
    epics_arch_inc_dir,
    psalg_inc_dir,
  ],
  dependencies: [
    psalg_utils_dep,
  ],
  install: true,
  install_rpath: libdir_install_rpath,
  link_with: [],
)
