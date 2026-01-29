
rogue_dep = cc.find_library('rogue-core')

hsd_sources = files(
  'AdcCore.cc',
  'AdcSync.cc',
  'Adt7411.cc',
  'ClkSynth.cc',
  #'DmaCore.cc',
  #'FlashController.cc',
  'FexCfg.cc',
  'FmcCore.cc',
  'FmcSpi.cc',
  'Histogram.cc',
  'I2cSwitch.cc',
  'Jesd204b.cc',
  'LocalCpld.cc',
  'Mmcm.cc',
  'Pgp2b.cc',
  'Pgp3.cc',
  'PhyCore.cc',
  'PvDef.cc',
  'QABase.cc',
  'Tps2481.cc',
  #'Validator.cc,
  'Fmc134Ctrl.cc',
  'Fmc134Cpld.cc',
  'Ad7291.cc',
  'OptFmc.cc',
  'Module134.cc',
  'Module126.cc',
  'ModuleBase.cc',
  'ChipAdcReg.cc'
)

hsd_inc_dir = include_directories('../..')

hsd_lib = shared_library('hsd',
  hsd_sources,
  include_directories: [hsd_inc_dir],
  dependencies: [
    psalg_utils_dep,
    threads_dep,
    #tr_dep,
  ],
  link_with: [
    mmhw_lib,
    service_lib,
  ],
  install_rpath: libdir_install_rpath,
  install: true,
)

mempeek = executable('mempeek',
  'mempeek.cc',
  include_directories: hsd_inc_dir,
  link_with: mmhw_lib,
  dependencies: [
    #rt_dep,
    #dl_dep,
  ],
  install_rpath: libdir_install_rpath,
  install: true,
)


hsd_eyescan = executable('hsd_eyescan',
  'hsd_eyescan.cc',
  include_directories: hsd_inc_dir,
  dependencies: [
    threads_dep,
    #rt_dep,
    #dl_dep,
  ],
  link_with: mmhw_lib,
  install_rpath: libdir_install_rpath,
  install: true,
)

hsd134PVs_sources = [
  'hsd134PVs.cc',
  'PV134Ctrls.cc',
  'PVCtrlsBase.cc',
  'PV134Stats.cc',
]

hsd134PVs = executable('hsd134PVs',
  hsd134PVs_sources,
  include_directories: hsd_inc_dir,
  dependencies: [
    python_dep,
    epics_dep,
    psalg_utils_dep,
    rogue_dep,
    threads_dep,
  ],
  link_with: [
    mmhw_lib,
    hsd_lib,
    epicstools_lib,
    service_lib,
    app_lib,
 ],
  install_rpath: libdir_install_rpath,
  install: true,
)

hsd126PVs_sources = [
  'hsd126PVs.cc',
  'PV126Ctrls.cc',
  'PVCtrlsBase.cc',
  'PV126Stats.cc',
]

hsd126PVs = executable('hsd126PVs',
  hsd126PVs_sources,
  include_directories: [hsd_inc_dir],
  dependencies: [
    epics_dep,
  ],
  link_with: [
    hsd_lib,
    mmhw_lib,
    epicstools_lib,
    service_lib,
    app_lib,
  ],
  install_rpath: libdir_install_rpath,
  install: true,
)

hsd_sim = executable('hsd_sim',
  'hsd_sim.cc', 
  install_rpath: libdir_install_rpath,
  install: true,
)

hsdRead = executable('hsdRead',
  [
    'hsdRead.cc',
    '../../../psalg/psalg/digitizer/src/Stream.cc'
  ],
  dependencies: [
    epics_dep,
    xtcdata_dep,
    threads_dep,
  ],
  include_directories: [
    hsd_inc_dir,
    include_directories('../../../psalg/')
    ],
  link_with: [
    epicstools_lib,
  ],
  install_rpath: libdir_install_rpath,
  install: true,
)
