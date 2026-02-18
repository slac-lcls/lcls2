numpy_dep = dependency('numpy')

psana_incdir = include_directories(
    '.'
)

py.extension_module(
    'dgram',
    'src/dgram.cc',
    include_directories: [
        psana_incdir,
        xtcdata_incdir
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [xtc_shlib],
    dependencies: [numpy_dep]
)

container_ext_include = include_directories(
    '../xtcdata/xtcdata/xtc'
)

py.extension_module(
    'container',
    'src/container.cc',
    include_directories: [
        psana_incdir,
        xtcdata_incdir
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [xtc_shlib],
    dependencies: [numpy_dep]
)

py.extension_module(
    'shmem',
    'psana/shmem/shmem.pyx',
    include_directories: [
        psana_incdir,
        xtcdata_incdir,
        psalg_incdir
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [xtc_shlib, psalg_shlib],
    dependencies: [numpy_dep],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'peakFinder',
    'psana/peakFinder/peakFinder.pyx',
    'psana/peakFinder/src/PeakFinderAlgos.cc',
    'psana/peakFinder/src/LocalExtrema.cc',
    include_directories: [
        psana_incdir,
        xtcdata_incdir,
        psalg_incdir
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [xtc_shlib, utils_shlib],
    dependencies: [numpy_dep],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'psalg_ext',
    'psana/peakFinder/psalg_ext.pyx',
    'psana/peakFinder/src/PeakFinderAlgosLCLS1.cc',
    'psana/peakFinder/src/LocalExtrema.cc',
    include_directories: [
        psana_incdir,
        xtcdata_incdir,
        psalg_incdir
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [xtc_shlib],
    dependencies: [numpy_dep],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'peakfinder8',
    'psana/peakFinder/peakfinder8.pyx',
    'psana/peakFinder/peakfinder8.cc',
    'psana/peakFinder/src/LocalExtrema.cc',
    include_directories: [
        psana_incdir,
        xtcdata_incdir,
        psalg_incdir
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [xtc_shlib],
    dependencies: [numpy_dep],
    override_options : ['cython_language=cpp'],
)


py.extension_module(
    'constFracDiscrim',
    'psana/constFracDiscrim/constFracDiscrim.pyx',
    'psana/constFracDiscrim/src/ConstFracDiscrim.cc',
    include_directories: [psana_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [],
    dependencies: [numpy_dep],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'dgramCreate',
    'psana/peakFinder/dgramCreate.pyx',
    include_directories: [
        psana_incdir,
        xtcdata_incdir
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [],
    dependencies: [numpy_dep],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'dgramchunk',
    'src/dgramchunk.pyx',
    include_directories: [psana_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [],
    dependencies: [],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'smdreader',
    'psana/smdreader.pyx',
    include_directories: [
        psana_incdir,
        xtcdata_incdir
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [xtc_shlib],
    dependencies: [numpy_dep],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'eventbuilder',
    'psana/eventbuilder.pyx',
    include_directories: [psana_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [],
    dependencies: [],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'parallelreader',
    'psana/parallelreader.pyx',
    include_directories: [psana_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [],
    dependencies: [],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'dgramedit',
    'psana/dgramedit.pyx',
    include_directories: [
        psana_incdir,
        xtcdata_incdir
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [],
    dependencies: [
        numpy_dep,
        xtcdata_dep,
    ],
    override_options : ['cython_language=cpp'],
 )

py.extension_module(
    'quadanode',
    'psana/quadanode.pyx',
    include_directories: [psana_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [],
    dependencies: [],
    override_options : ['cython_language=cpp'],
 )

py.extension_module(
    'dgramlite',
    'psana/dgramlite.pyx',
    include_directories: [
        psana_incdir,
        xtcdata_incdir
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [],
    dependencies: [],
    override_options : ['cython_language=cpp'],
 )

py.extension_module(
    'hsd',
    'psana/hsd/hsd.pyx',
    include_directories: [
        psana_incdir,
        psalg_incdir,
        xtcdata_incdir
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [],
    dependencies: [numpy_dep],
    override_options : ['cython_language=cpp'],
 )

py.extension_module(
    'ndarray',
    sources: ['psana/pycalgos/NDArray_ext.pyx',
              'psana/peakFinder/src/WFAlgos.cc'
            ],
    include_directories: [
        psana_incdir,
        psalg_incdir,
        xtcdata_incdir
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [],
    dependencies: [numpy_dep],
    override_options : ['cython_language=cpp'],
 )

py.extension_module(
    'utilsdetector_ext',
    sources: ['psana/pycalgos/utilsdetector_ext.pyx',
              'psana/pycalgos/UtilsDetector.cc'
            ],
    include_directories: [psana_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [],
    dependencies: [numpy_dep],
    override_options : ['cython_language=cpp'],
 )

py = import('python').find_installation()

install_subdir(
  'psana',
  install_dir: py.get_install_dir()
)