
psdaq_inc = include_directories('../../..')
drp_inc = include_directories('../../../drp')


pgpdriver_lib = shared_library('pgpdriver',
  [
    'pgpdriver.cc',
    'aximicronn25q.cc',
    'mcsfile.cc',
  ],
  include_directories: [
    psdaq_inc,
    drp_inc,
  ],
  link_with: epicstools_lib,
  install: true,
  cpp_args: [
    '-Wall',
    '-Wno-uninitialized',
  ],
)

#install_headers(
#  'pgpdriver.h'
#)

executable('status',
  'status.cc',
  include_directories: [
    psdaq_inc,
    drp_inc
  ],
  link_with: [pgpdriver_lib],
  install_rpath: libdir_install_rpath,
  #install: true,
)

executable('pgpmon',
  'pgpmon.cc',
  include_directories: [
    psdaq_inc,
    drp_inc
  ],
  link_with: [pgpdriver_lib],
  install_rpath: libdir_install_rpath,
  #install: true,
)

executable('looptest',
  'looptest.cc',
  include_directories: [
    psdaq_inc,
    drp_inc
  ],
  link_with: [pgpdriver_lib],
  install_rpath: libdir_install_rpath,
  #install: true,
)

executable('promload',
  'promload.cc',
  include_directories: [
    psdaq_inc,
    drp_inc
  ],
  link_with: [pgpdriver_lib],
  install_rpath: libdir_install_rpath,
  #install: true,
)

executable('mcstest',
  'mcstest.cc',
#    include_directories: [
#        psdaq_inc,
#        drp_inc
#    ],
  link_with: [pgpdriver_lib],
  install_rpath: libdir_install_rpath,
  #install: true,
)
