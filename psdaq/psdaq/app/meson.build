
app_incdir = include_directories('../..')

app_lib = shared_library('app',
  'AppUtils.cc',
  include_directories: app_incdir,
  dependencies: [
    threads_dep,
  ],
  install_rpath: libdir_install_rpath,
  install: true,
)

hpsBldServer = executable('hpsBldServer',
  'hpsBldServer.cc',
  include_directories: app_incdir,
  dependencies: [
    python_dep,
    threads_dep,
    #dl_dep,
    #rt_dep,
  ],
  link_with: [
    bld_lib,
    app_lib,
  ],
  install_rpath: libdir_install_rpath,
  install: true,
)


hpsBldClient = executable('hpsBldClient',
  'hpsBldClient.cc',
  include_directories: app_incdir,
  dependencies: [
    xtcdata_dep,
    epics_dep,
    #dl_dep,
    #rt_dep,
  ],
  link_with: [
    bld_lib,
    epicstools_lib,
    app_lib,
    kcu_lib,
  ],
  install_rpath: libdir_install_rpath,
  install: true,
)

hpsBldCopy = executable('hpsBldCopy',
  'hpsBldCopy.cc',
  include_directories: app_incdir,
  dependencies: [
    xtcdata_dep,
    epics_dep,
    #dl_dep,
    #rt_dep,
  ],
  link_with: [
    epicstools_lib,
    bld_lib,
    app_lib
  ],
  install_rpath: libdir_install_rpath,
  install: true
)

lclsBldServer = executable('lclsBldServer',
  'lclsBldServer.cc',
  include_directories: app_incdir,
  link_with: [
    tpr_lib,
    app_lib,
  ],
  dependencies: [
    #rt_dep,
    #dl_dep,
  ],
  install_rpath: libdir_install_rpath,
  install: true
)

tprtrig = executable('tprtrig',
  'tprtrig.cc',
  include_directories: app_incdir,
  link_with: tpr_lib,
  install_rpath: libdir_install_rpath,
  install: true,
  dependencies: [
    #rt_dep,
    #dl_dep,
  ],
)

tprioc = executable('tprioc',
  'tprca.cc',
  include_directories: app_incdir,
  dependencies: [
    epics_dep,
    #dl_dep,
    #rt_dep,
  ],
  link_with: [
    epicstools_lib,
    tpr_lib,
  ],
  install_rpath: libdir_install_rpath,
  install: true
)

tprsh = executable('tprsh',
  'tprsh.cc',
  include_directories: app_incdir,
  dependencies: [
    threads_dep,
  ],
  install_rpath: libdir_install_rpath,
  install: true,
)

tprutil = executable('tprutil',
  'tprutil.cc',
  include_directories: app_incdir,
  link_with: tpr_lib,
  install_rpath: libdir_install_rpath,
  install: true,
  dependencies: [
    #rt_dep,
    #dl_dep,
  ],
)

xcasttest = executable('xcasttest',
  'xcasttest.cc',
  include_directories: app_incdir,
  link_with: app_lib,
  install_rpath: libdir_install_rpath,
  install: true,
  dependencies: [
    #rt_dep,
    dl_dep,
  ],
)

pgpca = executable('pgpca',
  'pgpca.cc',
  include_directories: app_incdir,
  dependencies: [
    epics_dep,
    #dl_dep,
    #rt_dep,
  ],
  link_with: epicstools_lib,
  install_rpath: libdir_install_rpath,
  install: true,
)

pvcam = executable('pvcam',
  'pvcam.cc',
  include_directories: app_incdir,
  dependencies: [
    epics_dep,
    #dl_dep,
    #rt_dep,
  ],
  link_with: tpr_lib,
  install_rpath: libdir_install_rpath,
  install: true,
)

bldpayload = executable('bldpayload',
  'bldpayload.cc',
  include_directories: app_incdir,
  dependencies: 
  [
    epics_dep,
    xtcdata_dep,
    psalg_utils_dep,
    #dl_dep,
    #rt_dep,
  ],
  link_with: epicstools_lib,
  install_rpath: libdir_install_rpath,
  install: true,
)

sim_udpencoder = executable('sim_udpencoder',
  'sim_udpencoder.cc',
  include_directories: app_incdir,
  dependencies: 
  [
    xtcdata_dep,
    psalg_utils_dep,
    python_dep,
    #dl_dep,
    #rt_dep,
  ],
  link_with: tpr_lib,
  install_rpath: libdir_install_rpath,
  install: true,
)

json2xtc = executable('json2xtc',
  'json2xtc.cc',
  include_directories: [
    app_incdir,
  ],
  dependencies:
  [
    xtcdata_dep,
  ],
  link_with: [
    service_lib,
  ],
  install_rpath: libdir_install_rpath,
  install: true,
)
