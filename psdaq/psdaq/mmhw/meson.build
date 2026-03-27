
mmhw_inc_dir = include_directories('../..')

mmhw_lib = shared_library('mmhw',
  ['AxiVersion.cc',
    'HexFile.cc',
    'McsFile.cc',
    'Reg.cc',
    'Reg64.cc',
    'RegProxy.cc',
    'RingBuffer.cc',
    'GthEyeScan.cc',
    'TprCore.cc',
    'TriggerEventManager.cc',
    'Xvc.cc',
  ],
  include_directories: mmhw_inc_dir,
  link_with: service_lib,
  dependencies: [threads_dep],
  install: true,
)

mmregtest_exe = executable('mmregtest',
  'mmregtest.cc',
  include_directories: mmhw_inc_dir,
  #install: true,
)

mmxvc_exe = executable('mmxvc',
  'mmxvc.cc',
  include_directories: mmhw_inc_dir,
  link_with: [
    mmhw_lib,
    service_lib,
  ],
  dependencies: [
    threads_dep,
    python_dep,
  ],
  #install: true,
)