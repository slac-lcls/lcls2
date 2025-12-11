shmemsrv_sources = files(
  'src/XtcMonitorServer.cc',
  'src/XtcMonitorMsg.cc',
  'src/TransitionCache.cc',
  'src/ProcInfo.cc',
  'XtcRunSet.cc'
)

shmemcli_source = files(
  'src/ShmemClient.cc',
  'src/XtcMonitorMsg.cc'
)

shmem_headers = files(
    'DetectorTypes.hh',
    'Detector.hh',
    'DetectorStore.hh',
    'UtilsConfig.hh',
    'AreaDetector.hh',
    'AreaDetectorStore.hh',
    'AreaDetectorJungfrau.hh',
    'AreaDetectorPnccd.hh',
    'AreaDetectorOpal.hh',
    'AreaDetectorEpix100a.hh',
    'DataSourceSimulator.hh',


)

detector_incdir = include_directories(
    '../utils',
    '../calib',
    '../geometry',
    '../../../xtcdata/xtcdata/xtc'
)

detector_shlib = shared_library(
    'detector',
    detector_sources,
    include_directories: detector_incdir,
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib, utils_shlib, calib_shlib, geometry_shlib]
)

install_headers(
    detector_headers,
    install_dir: 'include/psalg/detector'
)






target_link_libraries(shmemcli
  xtcdata::xtc
  rt
)

target_link_libraries(shmemsrv
  xtcdata::xtc
)

add_executable(shmemClient
  shmemClientApp.cc
)

target_link_libraries(shmemClient
  xtcdata::xtc
  shmemcli
  rt
)

add_executable(shmemWriter
  shmemWriter.cc
)

target_link_libraries(shmemWriter
  xtcdata::xtc
  shmemcli
  rt
)

add_executable(shmemServer
  shmemServer.cc
)

target_link_libraries(shmemServer
  xtcdata::xtc
  shmemsrv
  rt
)

target_include_directories(shmemcli PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

install(FILES
  ShmemClient.hh
  XtcMonitorServer.hh
  XtcMonitorMsg.hh
  DESTINATION include/psalg/shmem
)

install(TARGETS shmemcli
  shmemClient shmemWriter
  EXPORT psalgTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS shmemsrv
  shmemServer
  EXPORT psalgTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
