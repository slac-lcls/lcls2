numpy_dep = dependency('numpy')

psana_incdir = include_directories(
    '.'
)

py.extension_module(
    'dgram',
    'src/dgram.cc',
    include_directories: [psana_incdir, xtcdata_incdir],
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
    include_directories: [psana_incdir, xtcdata_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [xtc_shlib],
    dependencies: [numpy_dep]
)

py.extension_module(
    'shmem',
    'psana/shmem/shmem.pyx',
    include_directories: [psana_incdir, xtcdata_incdir, psalg_incdir],
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
    include_directories: [psana_incdir, xtcdata_incdir, psalg_incdir],
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
    include_directories: [psana_incdir, xtcdata_incdir, psalg_incdir],
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
    include_directories: [psana_incdir, xtcdata_incdir, psalg_incdir],
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
    include_directories: [psana_incdir, xtcdata_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [],
    dependencies: [numpy_dep],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'psana.dgramchunk',
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
    'psana.smdreader',
    'psana/smdreader.pyx',
    include_directories: [psana_incdir, xtcdata_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [xtc_shlib],
    dependencies: [numpy_dep],
    override_options : ['cython_language=cpp'],
)

py.extension_module(
    'psana.eventbuilder',
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
    'psana.parallelreader',
    'psana/parallelreader.pyx',
    include_directories: [psana_incdir],
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [],
    dependencies: [],
    override_options : ['cython_language=cpp'],
)

 ext = Extension("psana.dgramedit",
                    sources=["psana/dgramedit.pyx"],
                    libraries = ['xtc'],
                    #include_dirs=["psana",np.get_include(), os.path.join(instdir, 'include')],
                    #library_dirs = [os.path.join(instdir, 'lib')],
                    include_dirs = ["psana", np.get_include(), os.path.join(instdir, 'include'), xtc_headers ],
                    library_dirs = [os.path.join(instdir, 'lib'), xtc_lib_path],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
    )
    CYTHON_EXTS.append(ext)

    ext = Extension("quadanode",
                    sources=["psana/quadanode.pyx"],
                    include_dirs=["psana",np.get_include(), os.path.join(instdir, 'include')],
                    library_dirs = [os.path.join(instdir, 'lib')],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
    )
    CYTHON_EXTS.append(ext)

    ext = Extension("psana.dgramlite",
                    sources=["psana/dgramlite.pyx"],
                    libraries = ['xtc'],
                    #include_dirs=["psana",np.get_include(), os.path.join(instdir, 'include')],
                    #iibrary_dirs = [os.path.join(instdir, 'lib')],
                    include_dirs = ["psana", np.get_include(), os.path.join(instdir, 'include'), xtc_headers ],
                    library_dirs = [os.path.join(instdir, 'lib'), xtc_lib_path],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
    )
    CYTHON_EXTS.append(ext)

#BUILD_LIST = ('PSANA','SHMEM','PEAKFINDER','HEXANODE','DGRAM','HSD','CFD','NDARRAY', 'PYCALGOS')# ,'XTCAV')
