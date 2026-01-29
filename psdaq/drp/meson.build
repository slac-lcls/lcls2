
drp_inc_dir = include_directories('..')
eigen_inc_dir = include_directories(join_paths(get_option('conda_prefix'), 'include', 'eigen3'))

sls_dep = cc.find_library('libSlsDetector')

drpbase_lib = shared_library('drpbase',
  [
    'PythonConfigScanner.cc',
    'BEBDetector.cc',
    'TTFex.cc',
    'XpmDetector.cc',
    'DrpBase.cc',
    'FileWriter.cc',
    'Si570.cc',
  ],
  include_directories: [drp_inc_dir],
  #install: true,
  #install_rpath: libdir_install_rpath,
  link_with: [
    mmhw_lib,
    epicstools_lib,
    contributor_lib,
    collection_lib,
    utilities_lib,
    service_lib,
    exporter_lib,
  ],
  dependencies: [
    python_dep,
    xtcdata_dep,
    psalg_utils_dep,
    epics_dep,
    threads_dep,
    prometheus_pull_dep,
    zmq_dep,
  ],
)

drp_dep = declare_dependency(
  link_with: drpbase_lib,
  include_directories: include_directories('..')
)

executable('drp',
  [
    'AreaDetector.cc',
    'Digitizer.cc',
    'EpixHR2x2.cc',
    'EpixHRemu.cc',
    'EpixM320.cc',
    'EpixUHR.cc',
    'Epix100.cc',
    'EpixQuad.cc',
    'Jungfrau.cc',
    'JungfrauDetectorId.cc',
    'JungfrauEmulator.cc',
    'TimingDef.cc',
    'TimingBEB.cc',
    'TimingSystem.cc',
    'TimeTool.cc',
    'Wave8.cc',
    'HREncoder.cc',
    'Opal.cc',
    'OpalTTFex.cc',
    'Piranha4.cc',
    'Piranha4TTFex.cc',
    'TebReceiver.cc',
    'PGPDetector.cc',
    'PGPDetectorApp.cc',
    'drp.cc',
  ],
  include_directories: [drp_inc_dir],
  dependencies: [
    xtcdata_dep,
    python_dep,
    psalg_utils_dep,
    epics_dep,
    zmq_dep,
    threads_dep,
    prometheus_pull_dep,
    sls_dep,
  ],
  install: true,
  install_rpath: libdir_install_rpath,
  link_with: [
    epicstools_lib,
    collection_lib,
    drpbase_lib,
    service_lib,
    exporter_lib,
    contributor_lib,
    utilities_lib,
    detector_shlib,
  ],
)

executable('drp_bld',
  [
    'TebReceiver.cc',
    #'BldDetectorSlow.cc',
    'BldDetector.cc',
    'BldNames.cc',
  ],
  include_directories: [drp_inc_dir],
  dependencies: [
    xtcdata_dep,
    python_dep,
    psalg_utils_dep,
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
    drpbase_lib,
    service_lib,
    exporter_lib,
    contributor_lib,
    utilities_lib,
  ],
)

executable('drp_pva',
  [
    'TebReceiver.cc',
    'PvaDetector.cc'
  ],
  include_directories: [drp_inc_dir],
  dependencies: [
    xtcdata_dep,
    python_dep,
    psalg_utils_dep,
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
    drpbase_lib,
    service_lib,
    exporter_lib,
    contributor_lib,
    utilities_lib,
  ],
)

executable('drp_udpencoder',
  [
    'TebReceiver.cc',
    'UdpEncoder.cc'
  ],
  include_directories: [drp_inc_dir, eigen_inc_dir],
  dependencies: [
    xtcdata_dep,
    python_dep,
    psalg_utils_dep,
    zmq_dep,
    threads_dep,
    prometheus_pull_dep
  ],
  install: true,
  install_rpath: libdir_install_rpath,
  link_with: [
    collection_lib,
    drpbase_lib,
    service_lib,
    exporter_lib,
    contributor_lib,
    utilities_lib,
  ],
)

executable('tpr_trigger',
  'TprTrigger.cc',
  include_directories: [drp_inc_dir],
  dependencies: [
    psalg_utils_dep,
    zmq_dep,
    threads_dep,
    rt_dep,
  ],
  install: true,
  install_rpath: libdir_install_rpath,
  link_with: [
    collection_lib,
    tpr_lib,
  ],
)

executable('configdb_obj_2xtc',
  'configdb_obj_2xtc.cc',
  include_directories: [drp_inc_dir],
  dependencies: [
    xtcdata_dep,
    python_dep,
  ],
  #install: true,
  #install_rpath: libdir_install_rpath,
  link_with: [
    service_lib,
  ],
)

executable('pgpread',
  'pgpread.cc',
  include_directories: [drp_inc_dir],
  dependencies: [
    xtcdata_dep,
    psalg_utils_dep,
  ],
  #install: true,
  #install_rpath: libdir_install_rpath,
  link_with: [
    mmhw_lib,
  ],
)

executable('pgpread_timetool',
  [
    'AxiBatcherParser.cc',
    'pgpread_timetool.cc'
  ],
  include_directories: [drp_inc_dir],
  dependencies: [
    xtcdata_dep,
    python_dep,
    rt_dep,
  ],
  #install: true,
  #install_rpath: libdir_install_rpath,
  link_with: [
    service_lib,
  ],
)

executable('pgpread_jungfrau',
  'pgpread_jungfrau.cc',
  include_directories: [drp_inc_dir],
  dependencies: [
    xtcdata_dep,
    psalg_utils_dep,
  ],
  #install: true,
  #install_rpath: libdir_install_rpath,
  link_with: [
    mmhw_lib,
  ],
)

executable('pgpread_epixM320mon',
  'pgpread_epixM320mon.cc',
  include_directories: [drp_inc_dir],
  dependencies: [],
  #install: true,
  #install_rpath: libdir_install_rpath,
  link_with: [],
)

executable('AxiBatcherParserTest',
  [
    'AxiBatcherParserTest.cc',
    'AxiBatcherParser.cc'
  ],
  include_directories: [drp_inc_dir],
  dependencies: [
    xtcdata_dep,
    python_dep,
  ],
  #install: true,
  #install_rpath: libdir_install_rpath,
  link_with: [],
)

executable('drp_validate',
  'validate.cc',
  include_directories: [drp_inc_dir],
  dependencies: [
    threads_dep,
  ],
  #install: true,
  #install_rpath: libdir_install_rpath,
  link_with: [],
)

executable('opaltt_test',
  'opaltt_test.cc',
  include_directories: [drp_inc_dir],
  dependencies: [
    xtcdata_dep,
    psalg_utils_dep,
  ],
  #install: true,
  #install_rpath: libdir_install_rpath,
  link_with: [],
)

executable('fileWriteTest',
  'fileWriteTest.cc',
  include_directories: [drp_inc_dir],
  dependencies: [
    xtcdata_dep,
    psalg_utils_dep,
  ],
  #install: true,
  #install_rpath: libdir_install_rpath,
  link_with: [
    drpbase_lib,
  ],
)

executable('drp_groupsync',
  'groupsync.cc',
  include_directories: [drp_inc_dir],
  dependencies: [
    threads_dep,
  ],
  #install: true,
  #install_rpath: libdir_install_rpath,
  link_with: [],
)

executable('ePixMDescTest',
  'ePixMDescTest.cc',
  include_directories: [drp_inc_dir],
  dependencies: [],
  #install: true,
  #install_rpath: libdir_install_rpath,
  link_with: [],
)

executable('jungfrauTest',
  [
    'jungfrauTest.cc',
    'JungfrauDetectorId.cc'
  ],
  include_directories: [drp_inc_dir],
  dependencies: [
    sls_dep,
  ],
  #install: true,
  #install_rpath: libdir_install_rpath,
  link_with: [],
)

executable('jungfrauSim',
  'jungfrauSim.cc',
  include_directories: [drp_inc_dir],
  dependencies: [
    xtcdata_dep,
  ],
  #install: true,
  #install_rpath: libdir_install_rpath,
  link_with: [],
)
