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
    link_with: [xtc_shlib, psalg_shlib, shmemclient_shlib],
    dependencies: [numpy_dep],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'peakFinder_ext',
    'psana/peakFinder/peakFinder_ext.pyx',
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

roentdek_dep = cc.find_library('Resort64c_x64', required: false)
hexanode_incdir = include_directories('psana/hexanode')

if roentdek_dep.found()
  py.extension_module(
    'hexanode_ext',
    sources: ['psana/hexanode/hexanode_ext.pyx',
              'psana/hexanode/src/cfib.cc',
              'psana/hexanode/src/wrap_resort64c.cc',
              'psana/hexanode/src/SortUtils.cc',
              'psana/hexanode/src/LMF_IO.cc',
    ],
    include_directories: [psana_incdir, hexanode_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [],
    dependencies: [numpy_dep, roentdek_dep],
    override_options : ['cython_language=cpp'],
  )
endif

install_subdir(
  'psana',
  install_dir: py.get_install_dir(),
  exclude_directories: [
    'graphqt/lcls1',
    'pscalib/dcs/lcls1',
    'pscalib/calib/lcls1',
    'xtcav/samples',
    'xtcav/examples',
    'pyalgos/app/lcls1',
    'pyalgos/fiber/lcls1',
    'hexanode/examples/lcls1',
    'pyalgos/generic/lcls1',
    'pscalib/examples',
    'hsd',
    'constFracDiscrim',
    #'pycalgos',
    'shmem',
    'tests',
  ],
  exclude_files: [
    'graphqt/QWDirNameV2.py',
    'peakFinder/examples/lcls1-ex-05-pf-on-data.py',
    'peakFinder/examples/lcls1-ex-05-pf-on-data-cspad-mask.py',
    'hexanode/examples/conv-01-acqiris-xtc-to-h5.py',
    'hexanode/examples/depr-conv-01-xtc1-to-json.py',
    'dgramedit.pxd',
    'dgramedit.pyx',
    'dgramlite.pxd',
    'dgramlite.pyx',
    'eventbuilder.pyx',
    'mman_compat.pxd',
    'parallelreader.pxd',
    'parallelreader.pyx',
    'quadanode.pyx',
    'smdreader.pyx',
    'cydebug.pxh',
  ],
)

# Copy geometry files over
install_subdir(
  'psana/pscalib/geometry/data',
  install_dir: py.get_install_dir() / 'psana/pscalib/geometry',
)

# Copy graphqt icons etc
install_subdir(
  'psana/graphqt/data/icons',
  install_dir: py.get_install_dir() / 'psana/graphqt/data',
)
