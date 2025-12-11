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
    install: true,
    install_rpath: libdir_install_rpath
)

install_headers(
    utils_headers,
    install_dir: 'include/psalg/utils'
)
