
com_dep = cc.find_library('Com', dirs: [join_paths(get_option('epics_base'), 'lib', get_option('epics_host_arch'))])
ca_dep = cc.find_library('ca', dirs: [join_paths(get_option('epics_base'), 'lib', get_option('epics_host_arch'))])
pvaClient_dep = cc.find_library('pvaClient', dirs: [join_paths(get_option('epics_base'), 'lib', get_option('epics_host_arch'))])
pvAccessCA_dep = cc.find_library('pvAccessCA', dirs: [join_paths(get_option('epics_base'), 'lib', get_option('epics_host_arch'))])
pvaData_dep = cc.find_library('pvData', dirs: [join_paths(get_option('epics_base'), 'lib', get_option('epics_host_arch'))])
pvAccess_dep = cc.find_library('pvAccess', dirs: [join_paths(get_option('epics_base'), 'lib', get_option('epics_host_arch'))])

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
