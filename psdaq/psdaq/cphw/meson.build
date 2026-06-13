
cphw_lib = shared_library('cphw',
  ['Reg_rssi.cc', 'AmcPLL.cc', 'AmcTiming.cc', 'AxiVersion.cc',
   'GthEyeScan.cc', 'GthRxAlign.cc', 'HsRepeater.cc', 'Logging.cc',
   'RingBuffer.cc', 'TimingRx.cc', 'XBar.cc', 'BldControl.cc', 'Xvc.cc'
  ],
  include_directories: include_directories('../..'),
  install: true
)

cphwr_lib = shared_library('cphwr',
  ['Reg.cc', 'AmcPLL.cc', 'AmcTiming.cc', 'AxiVersion.cc',
   'GthEyeScan.cc', 'GthRxAlign.cc', 'HsRepeater.cc', 'Logging.cc',
   'RingBuffer.cc', 'TimingRx.cc', 'XBar.cc', 'BldControl.cc', 'Xvc.cc'
  ],
  include_directories: include_directories('../..'),
  install: true
)

amc_timing_udp_exe = executable('amc_timing_udp',
  'amc_timing.cc',
  link_with: cphwr_lib,
  dependencies: [threads_dep],
  include_directories: include_directories('../..'),
  install: true
)

amc_eyescan_exe = executable('amc_eyescan',
  'amc_eyescan.cc',
  link_with: cphwr_lib,
  dependencies: [threads_dep],
  include_directories: include_directories('../..'),
  install: true
)

hsrCfg_exe = executable('hsrCfg',
  'hsrCfg.cc',
  link_with: cphwr_lib,
  dependencies: [threads_dep],
  include_directories: include_directories('../..'),
  install: true
)

regtest_exe = executable('regtest',
  'regtest.cc',
  link_with: cphwr_lib,
  dependencies: [threads_dep],
  include_directories: include_directories('../..'),
  install: true
)

xvc_srv_exe = executable('xvc_srv',
  'xvc_srv.cc',
  link_with: cphwr_lib,
  dependencies: [threads_dep],
  include_directories: include_directories('../..'),
  install: true
)
