
threads_dep = dependency('threads')
dl_dep  = meson.get_compiler('cpp').find_library('dl')
rt_dep  = meson.get_compiler('cpp').find_library('rt')
yaml_cpp_dep = meson.get_compiler('cpp').find_library('yaml-cpp')


xpm_sources = files(
    'PVSeq.cc',
    'PVStats.cc',
    'PVPStats.cc',
    'PVCtrls.cc',
    'PVPCtrls.cc',
    'PVLinkCtrls.cc',
    'user_sequence.cc',
    'XpmSequenceEngine.cc',
    'Module.cc'
)

xpm_lib = shared_library(
    'xpm',
    xpm_sources,
    dependencies: [
        yaml_cpp_dep,
        xtcdata_dep
    ],
    include_directories: include_directories('..'),
    link_with: epicstools_lib,
    install: true,
)

executable('xpm_simple',
    'xpm_simple.cc',
    dependencies : [
        yaml_cpp_dep,
        dl_dep,
        threads_dep,
        rt_dep
    ],
    link_with: [service_lib, cphw_lib],
    install : true
)
