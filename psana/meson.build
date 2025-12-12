numpy_dep = dependency('numpy')

dgram_ext_include = include_directories(
    '../xtcdata/xtcdata/xtc'
)

py.extension_module(
    'dgram',
    'src/dgram.cc',
    include_directories: dgram_ext_include,
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
    include_directories: container_ext_include,
    install: true,
    install_rpath: libdir_install_rpath,
    subdir: 'psana',
    link_with: [xtc_shlib],
    dependencies: [numpy_dep]
)
