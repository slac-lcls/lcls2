
service_sources = files(
    'SysClk.cc',
    'Semaphore.cc',
    'SemLock.cc',
    'Lock.cc',
    'Pool.cc',
    'Task.cc',
    'TaskObject.cc',
    'Timer.cc',
    'GenericPool.cc',
    'GenericPoolW.cc',
    'Histogram.cc',
    'Dl.cc',
    'Json2Xtc.cc',
    'IpcUtils.cc'
)

service_lib = shared_library('service',
    service_sources,
    dependencies: [
        python_dep,
        threads_dep,
        xtcdata_dep,
        dl_dep,
        rt_dep,
        psalg_utils_dep,
    ],
    include_directories: [
        include_directories('..'),
    ],
    install_rpath: libdir_install_rpath,
    install: true,
)

collection_sources = files(
    'Collection.cc',
    'kwargs.cc',
    'range.cc',
)

collection_lib = shared_library('collection',
    collection_sources,
    dependencies: [
        psalg_utils_dep,
        zmq_dep,
        nlohmann_json_dep,
    ],
    install: true,
    install_rpath: libdir_install_rpath,
)

exporter_sources = files(
  'MetricExporter.cc',
)

exporter_lib = shared_library('exporter',
    exporter_sources,
    dependencies: [
        psalg_utils_dep,
        prometheus_core_dep,
        prometheus_pull_dep
    ],
    include_directories: include_directories('..'),
    install: true,
)

install_headers(
  'EbDgram.hh',
  subdir: 'psdaq/service',
)
