threads_dep = dependency('threads')
nlohmann_json_dep = dependency('nlohmann_json', required: false)

zmq_dep = cc.find_library('zmq')
dl_dep = cc.find_library('dl', required: false)
rt_dep = cc.find_library('rt', required: false)

prometheus_core_dep = cc.find_library('prometheus-cpp-core', 
  dirs: [
    join_paths(get_option('conda_prefix'), 'lib'), # python 3.11
    join_paths(get_option('conda_prefix'), 'lib64'), # python 3.9
  ]
)
prometheus_pull_dep = cc.find_library('prometheus-cpp-pull',
  dirs: [
    join_paths(get_option('conda_prefix'), 'lib'), # python 3.11
    join_paths(get_option('conda_prefix'), 'lib64'), # python 3.9
  ]
)

psdaq_incdir = include_directories('.')

subdir('psdaq')
subdir('drp')
#subdir('drpGpu') # Only compiles in Python 3.11
subdir('epicsArch')

py.extension_module(
    'EbDgram',
    'psdaq/trigger/EbDgram.pyx',
    include_directories: [
        psdaq_incdir,
        xtcdata_incdir,
    ],
    install: true,
    subdir: 'psdaq',
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib],
    override_options: ['cython_language=cpp'],
)

py.extension_module(
    'ResultDgram',
    'psdaq/trigger/ResultDgram.pyx',
    include_directories: [
        psdaq_incdir,
        xtcdata_incdir,
        eb_incdir,
    ],
    install: true,
    subdir: 'psdaq',
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib],
    override_options: ['cython_language=cpp'],
)

py.extension_module(
    'TmoTebData',
    'psdaq/trigger/TmoTebData.pyx',
    include_directories: [
        psdaq_incdir,
        xtcdata_incdir,
    ],
    install: true,
    subdir: 'psdaq',
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib],
    override_options: ['cython_language=cpp'],
)

py.extension_module(
    'TimingTebData',
    'psdaq/trigger/TimingTebData.pyx',
    include_directories: [
        psdaq_incdir,
        xtcdata_incdir,
    ],
    install: true,
    subdir: 'psdaq',
    install_rpath: libdir_install_rpath,
    link_with: [xtc_shlib],
    override_options: ['cython_language=cpp'],
)

install_subdir(
  'psdaq',
  install_dir: py.get_install_dir(),
  exclude_directories: [
    'aes-stream-drivers/',
    'app/',
    'bld/',
    'cnf/',
    'cphw/',
    'dti/',
    #'epicstools/',
    'eyediag/',
    'mmhw/',
    'monreq/',
    'pgp/',
    #'seq/',
    'service/',
    'tests/',
    'tpr/',
    'xpm/',
    'hsd/src',
    'trigger/src',
    'eb/src',
  ],
  exclude_files: [
    'CMakeLists.txt',
    'meson.build',
    'trigger/meson.build',
    'trigger/EbDgram.pyx',
    'trigger/EbDgramw.pxd',
    'trigger/ResultDgram.pyx',
    'trigger/ResultDgramw.pxd',
    'trigger/TimingTebData.pyx',
    'trigger/TimingTebDataw.pxd',
    'trigger/TmoTebData.pyx',
    'trigger/TmoTebDataw.pxd',
    'eb/meson.build',
    'slurm/readme.md',
    'slurm/slurm.cnf',
    'slurm/xthi.c',
    'hsd/meson.build',
  ],
)
