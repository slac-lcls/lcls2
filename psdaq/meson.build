numpy_dep = dependency('numpy')

psdaq_incdir = include_directories(
    '.'
)
# Check for CUDA 

#cuda = meson.get_compiler('cuda', required : false)
#if cuda.found()
#  ccbin = run_command('sh', '-c', 'echo $CC', check : false).stdout().strip()
#  if ccbin != ''
#    add_project_arguments(
#      ['-ccbin', ccbin],
#      language : 'cuda'
#    )
#  endif
#endif

subdir('psdaq')

py.extension_module(
    'EbDgram',
    'psdaq/trigger/EbDgram.pyx',
    include_directories: [
        psdaq_incdir,
        xtcdata_incdir
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
        xtcdata_incdir
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
        xtcdata_incdir
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
        xtcdata_incdir
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psdaq',
    link_with: [xtc_shlib],
    override_options : ['cython_language=cpp'],
)
