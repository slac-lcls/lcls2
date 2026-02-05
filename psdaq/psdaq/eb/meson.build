
eb_incdir = include_directories('..')

libfabric_dep = cc.find_library('libfabric')

service_dep = declare_dependency(
  link_with: service_lib,
  include_directories: include_directories('../..')
)

utilities_lib = shared_library('utilities',
  [
    'utilities.cc',
    'Endpoint.cc',
    'EbLfLink.cc',
    'EbLfServer.cc',
    'EbLfClient.cc',
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
    'EbCtrbInBase.cc',
    'TebContributor.cc',
    'MebContributor.cc',
    'BatchManager.cc',
    'Batch.cc',
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
    'EbAppBase.cc',
    'EventBuilder.cc',
    'EbEpoch.cc',
    'EbEvent.cc',
    'BatchManager.cc',
    'Batch.cc',
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
  'tstEbLfLink.cc',
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
)

teb_exe = executable('teb',
  'teb.cc',
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
)

tstClocks_exe = executable('tstClocks',
  'tstClocks.cc',
  include_directories: [
    eb_incdir,
    include_directories('../..'),
  ],
  dependencies: [
    #rt_dep,
  ],
  #install: true,
)

install_headers(
  [
    'eb.hh',
    'ResultDgram.hh',
  ],
  subdir: 'psdaq/eb'
)