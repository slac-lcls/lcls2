
shmemsrv_sources = files(
    'src/XtcMonitorServer.cc',
    'src/XtcMonitorMsg.cc',
    'src/TransitionCache.cc',
    'src/ProcInfo.cc',
    'src/XtcRunSet.cc'
)

shmemclient_sources = files(
    'src/ShmemClient.cc',
    'src/XtcMonitorMsg.cc'
)

shmem_headers = files(
    'ShmemClient.hh',
    'XtcMonitorServer.hh',
    'XtcMonitorMsg.hh',
    'TransitionCache.hh',
    'XtcRunSet.hh',
    'ProcInfo.hh'
)

shmemserver_shlib = shared_library(
    'shmemsrv',
    shmemsrv_sources,
    include_directories: [psalg_incdir, xtcdata_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib],
    dependencies: [threads_dep, rt_dep]
)

shmemclient_shlib = shared_library(
    'shmemcli',
    shmemclient_sources,
    include_directories: [psalg_incdir, xtcdata_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib],
    dependencies: [rt_dep]
)

executable(
    'shmemClient',
    'src/shmemClientApp.cc',
    install: true,
    include_directories: [psalg_incdir, xtcdata_incdir],
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib, shmemclient_shlib],
    dependencies: [rt_dep],
)

executable(
    'shmemWriter',
    'src/shmemWriter.cc',
    install: true,
    include_directories: [psalg_incdir, xtcdata_incdir],
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib, shmemclient_shlib],
    dependencies: [rt_dep],
)

executable(
    'shmemServer',
    'src/shmemServer.cc',
    install: true,
    include_directories: [psalg_incdir, xtcdata_incdir],
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib, shmemserver_shlib],
    dependencies: [rt_dep],
)

install_headers(
  [
    'ShmemClient.hh',
    'XtcMonitorServer.hh',
    'XtcMonitorMsg.hh',
  ],
    install_dir: 'include/psalg/shmem'
)
