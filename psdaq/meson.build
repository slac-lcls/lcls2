numpy_dep = dependency('numpy')
threads_dep = dependency('threads')
nlohmann_json_dep = dependency('nlohmann_json', required: false)

cc = meson.get_compiler('cpp')
zmq_dep = cc.find_library('zmq')
dl_dep = cc.find_library('dl', required: false)
rt_dep = cc.find_library('rt', required: false)

prometheus_core_dep = cc.find_library('prometheus-cpp-core', dirs: [join_paths(get_option('conda_prefix'), 'lib')])
prometheus_pull_dep = cc.find_library('prometheus-cpp-pull', dirs: [join_paths(get_option('conda_prefix'), 'lib')])

cuda_compiler = find_program('nvcc', required: false)

psdaq_incdir = include_directories('.')

subdir('psdaq')
subdir('drp')
subdir('drpGpu')
subdir('epicsArch')

py.extension_module(
    'EbDgram',
    'psdaq/trigger/EbDgram.pyx',
    include_directories: [
        psdaq_incdir,
        xtcdata_incdir,
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psdaq',
    link_with: [xtc_shlib],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'ResultDgram',
    'psdaq/trigger/ResultDgram.pyx',
    include_directories: [
        psdaq_incdir,
        xtcdata_incdir,
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psdaq',
    link_with: [xtc_shlib],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'TmoTebData',
    'psdaq/trigger/TmoTebData.pyx',
    include_directories: [
        psdaq_incdir,
        xtcdata_incdir,
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psdaq',
    link_with: [xtc_shlib],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'TimingTebData',
    'psdaq/trigger/TimingTebData.pyx',
    include_directories: [
        psdaq_incdir,
        xtcdata_incdir,
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psdaq',
    link_with: [xtc_shlib],
    override_options : ['cython_language=cpp'],
)
