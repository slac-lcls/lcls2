
com_dep = cc.find_library('Com', dirs: [epics_lib])
ca_dep = cc.find_library('ca', dirs: [epics_lib])
pvaClient_dep = cc.find_library('pvaClient', dirs: [epics_lib])
pvAccessCA_dep = cc.find_library('pvAccessCA', dirs: [epics_lib])
pvaData_dep = cc.find_library('pvData', dirs: [epics_lib])
pvAccess_dep = cc.find_library('pvAccess', dirs: [epics_lib])

epics_inc_dir = include_directories(join_paths(get_option('epics_base'), 'include'))
epics_comp_inc_dir = include_directories(join_paths(get_option('epics_base'), 'include', 'compiler', 'gcc'))
epics_osd_inc_dir = include_directories(join_paths(get_option('epics_base'), 'include', 'os', 'Linux'))


epics_dep = declare_dependency(
  include_directories: [
    epics_inc_dir,
    epics_osd_inc_dir,
    epics_comp_inc_dir,
  ],
  dependencies: [
    pvaClient_dep,
    pvaData_dep,
    pvAccess_dep,
    com_dep,
    pvAccessCA_dep,
    ca_dep,
  ],
)

epicstools_lib = shared_library('epicstools',
  [
    'EpicsProviders.cc',
    'EpicsPVA.cc',
    'MonTracker.cc',
    'PvServer.cc',
    'PvMonitorBase.cc',
  ],
  include_directories: include_directories('../..'),
  dependencies: [
    epics_dep,
    psalg_utils_dep,
    xtcdata_dep,
  ],
  install_rpath: libdir_install_rpath,
  install: true,
)

monExample_exe = executable('monExample',
  'monExample.cc',
  dependencies: epics_dep,
  link_with: epicstools_lib,
  install_rpath: libdir_install_rpath,
  #install: true,
)

tstPvMon_exe = executable('tstPvMon',
  'tstPvMon.cc',
  include_directories: include_directories('../..'),
  dependencies: [
    epics_dep,
    xtcdata_dep,
  ],
  link_with: epicstools_lib,
  install_rpath: libdir_install_rpath,
  #install: true,
)

