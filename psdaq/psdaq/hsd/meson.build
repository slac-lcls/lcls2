rogue_dep = cc.find_library('rogue-core')

hsd_inc_dir = include_directories('../..')

hsd_sources = files(
  'src/AdcCore.cc',
  'src/AdcSync.cc',
  'src/Adt7411.cc',
  'src/ClkSynth.cc',
  #'src/DmaCore.cc',
  #'src/FlashController.cc',
  'src/FexCfg.cc',
  'src/FmcCore.cc',
  'src/FmcSpi.cc',
  'src/Histogram.cc',
  'src/I2cSwitch.cc',
  'src/Jesd204b.cc',
  'src/LocalCpld.cc',
  'src/Mmcm.cc',
  'src/Pgp2b.cc',
  'src/Pgp3.cc',
  'src/PhyCore.cc',
  'src/PvDef.cc',
  'src/QABase.cc',
  'src/Tps2481.cc',
  #'src/Validator.cc,
  'src/Fmc134Ctrl.cc',
  'src/Fmc134Cpld.cc',
  'src/Ad7291.cc',
  'src/OptFmc.cc',
  'src/Module134.cc',
  'src/Module126.cc',
  'src/ModuleBase.cc',
  'src/ChipAdcReg.cc'
)

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
  cpp_args: ['-Wno-uninitialized',],
)

mempeek = executable('mempeek',
  'src/mempeek.cc',
  include_directories: hsd_inc_dir,
  link_with: mmhw_lib,
  dependencies: [
    #rt_dep,
    #dl_dep,
  ],
  install_rpath: libdir_install_rpath,
  #install: true,
)


hsd_eyescan = executable('hsd_eyescan',
  'src/hsd_eyescan.cc',
  include_directories: hsd_inc_dir,
  dependencies: [
    threads_dep,
    #rt_dep,
    #dl_dep,
  ],
  link_with: mmhw_lib,
  install_rpath: libdir_install_rpath,
  #install: true,
)

hsd134PVs_sources = [
  'src/hsd134PVs.cc',
  'src/PV134Ctrls.cc',
  'src/PVCtrlsBase.cc',
  'src/PV134Stats.cc',
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
  'src/hsd126PVs.cc',
  'src/PV126Ctrls.cc',
  'src/PVCtrlsBase.cc',
  'src/PV126Stats.cc',
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
  'src/hsd_sim.cc', 
  install_rpath: libdir_install_rpath,
  #install: true,
)

hsdRead = executable('hsdRead',
  [
    'src/hsdRead.cc',
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
  #install: true,
)
