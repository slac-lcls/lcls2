

kcu_inc_dir = include_directories('../../../..')
#rapidjson_inc_dir = include_directories(get_option('rapidjson_include'))  # adjust as needed

kcu_sources = [
    'Reg.cc',
    'GthEyeScan.cc',
    'Client.cc',
    'Si570.cc'
  ]

kcu_lib = shared_library('kcu',
  kcu_sources,
  include_directories: kcu_inc_dir,
  dependencies: xtcdata_dep,
  link_with: mmhw_lib,
  install_rpath: libdir_install_rpath,
  install: true,
)

static_kcu_lib = static_library('staticKcu',
  kcu_sources,
  dependencies: xtcdata_dep,
  include_directories: [kcu_inc_dir],
  link_with: [xtc_static_lib],
)

kcu_status = executable('kcuStatus',
  'xStatus.cpp',
  include_directories: [kcu_inc_dir],
  link_with: [static_kcu_lib],
  install_rpath: libdir_install_rpath,
  install: true,
)

kcu_pgp_status = executable('kcuPgpStatus',
  'xPgpStatus.cpp',
  include_directories: [kcu_inc_dir],
  link_with: [kcu_lib],
)

kcu_hist = executable('kcuHist',
  'xHist.cpp',
  include_directories: [kcu_inc_dir],
  install_rpath: libdir_install_rpath,
  install: true,
)

kcu_sim = executable('kcuSim',
  'xSim.cpp',
  include_directories: [kcu_inc_dir],
  install_rpath: libdir_install_rpath,
  install: true,
)

kcu_sim_valid = executable('kcuSimValid',
  'xSimValid.cpp',
  include_directories: [kcu_inc_dir],
  install_rpath: libdir_install_rpath,
  install: true,
)

kcu_loop_test = executable('kcuLoopTest',
  'xLoopTest.cpp',
  include_directories: [kcu_inc_dir],
)

kcu_reg = executable('kcuReg',
  'xReg.cpp',
  include_directories: [kcu_inc_dir],
)

kcu_read = executable('kcuRead',
  'xRead.cpp',
  include_directories: [kcu_inc_dir],
  dependencies: threads_dep,
)

kcu_read_bulk = executable('kcuReadBulk',
  'xReadBulk.cpp',
  include_directories: [kcu_inc_dir],
  dependencies: [threads_dep],
)

kcu_write = executable('kcuWrite',
  'xWrite.cpp',
  include_directories: [kcu_inc_dir],
  dependencies: [threads_dep],
)

kcu_dma_status = executable('kcuDmaStatus',
  'xDmaStatus.cpp',
  include_directories: [kcu_inc_dir],
)

kcu_margin = executable('kcuMargin',
  'xMargin.cpp',
  include_directories: [kcu_inc_dir],
  link_with: [kcu_lib],
  dependencies: [threads_dep],
)
