utils_sources = files(
    'src/Logger.cc',
    'src/DirFileIterator.cc',
    'src/trim.cc'
)

utils_headers = files(
    'Logger.hh',
    'Utils.hh',
    'MacTimeFix.hh',
    'DirFileIterator.hh',
    'ctest_utils.hh',
    'SysLog.hh',
    'NDArrayGenerators.hh',
    'trim.hh'
)
 
utils_shlib = shared_library(
    'utils',
    utils_sources,
    include_directories: psalg_incdir,
    install: true,
    install_rpath: libdir_install_rpath
)

psalg_utils_dep = declare_dependency(
    link_with: utils_shlib,
    include_directories: psalg_incdir
)

install_headers(
    utils_headers,
    install_dir: 'include/psalg/utils'
)
