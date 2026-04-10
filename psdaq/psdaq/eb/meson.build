eb_incdir = include_directories('..')

libfabric_dep = cc.find_library('fabric', dirs: [join_paths(get_option('conda_prefix'), 'lib')])

service_dep = declare_dependency(
  link_with: service_lib,
  include_directories: include_directories('../..')
)

utilities_lib = shared_library('utilities',
  [
    'src/utilities.cc',
    'src/Endpoint.cc',
    'src/EbLfLink.cc',
    'src/EbLfServer.cc',
    'src/EbLfClient.cc',
  ],
  include_directories: [eb_incdir],
  dependencies: [
    psalg_utils_dep,
    libfabric_dep,
    threads_dep,
    python_dep, 
    service_dep,
  ],
  install: true,
)

contributor_lib = shared_library('contributor',
  [
    'src/EbCtrbInBase.cc',
    'src/TebContributor.cc',
    'src/MebContributor.cc',
    'src/BatchManager.cc',
    'src/Batch.cc',
  ],
  dependencies: [
    libfabric_dep,
    psalg_utils_dep,
    xtcdata_dep,
    service_dep,
    threads_dep,
  ],
  link_with: [utilities_lib, exporter_lib],
  include_directories: [eb_incdir],
  install: true,
)

eventBuilder_lib = shared_library('eventBuilder',
  [
    'src/EbAppBase.cc',
    'src/EventBuilder.cc',
    'src/EbEpoch.cc',
    'src/EbEvent.cc',
    'src/BatchManager.cc',
    'src/Batch.cc',
  ],
  dependencies: [
    xtcdata_dep,
    psalg_utils_dep,
    zmq_dep,
  ],
  include_directories: [
    eb_incdir,
    include_directories('../..')
  ],
  link_with: [
    utilities_lib,
    service_lib,
    collection_lib,
    exporter_lib,
  ],
  install: true,
)

tstEbLfLink_exe = executable('tstEbLfLink',
  'src/tstEbLfLink.cc',
  include_directories: [
    eb_incdir,
    include_directories('../..'),
  ],
  dependencies: [
    psalg_utils_dep,
    xtcdata_dep,
    #rt_dep,
  ],
  link_with: [
    utilities_lib,
    collection_lib,
  ],
  #install: true,
  install_rpath: libdir_install_rpath,
)

teb_exe = executable('teb',
  'src/teb.cc',
  include_directories: [
    eb_incdir,
    include_directories('../..'),
  ],
  dependencies: [
    xtcdata_dep,
    psalg_utils_dep,
    python_dep,
    prometheus_pull_dep,
    zmq_dep,
    threads_dep,
    #rt_dep,
    #dl_dep,
  ],
  link_with: [
    collection_lib,
    eventBuilder_lib,
    service_lib,
    utilities_lib,
    exporter_lib,
  ],
  install: true,
  install_rpath: libdir_install_rpath,
)

tstClocks_exe = executable('tstClocks',
  'src/tstClocks.cc',
  include_directories: [
    eb_incdir,
    include_directories('../..'),
  ],
  dependencies: [
    #rt_dep,
  ],
  #install: true,
  install_rpath: libdir_install_rpath,
)

#install_headers(
#  [
#    'eb.hh',
#    'ResultDgram.hh',
#  ],
#  subdir: 'psdaq/eb'
#)
