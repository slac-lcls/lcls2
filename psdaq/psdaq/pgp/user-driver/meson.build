
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

install_headers(
  'pgpdriver.h'
)

executable('status',
  'status.cc',
  include_directories: [
    psdaq_inc,
    drp_inc
  ],
  link_with: [pgpdriver_lib],
)

executable('pgpmon',
  'pgpmon.cc',
  include_directories: [
    psdaq_inc,
    drp_inc
  ],
  link_with: [pgpdriver_lib],
)

executable('looptest',
  'looptest.cc',
  include_directories: [
    psdaq_inc,
    drp_inc
  ],
  link_with: [pgpdriver_lib],
)

executable('promload',
  'promload.cc',
  include_directories: [
    psdaq_inc,
    drp_inc
  ],
  link_with: [pgpdriver_lib],
)

executable('mcstest',
  'mcstest.cc',
#    include_directories: [
#        psdaq_inc,
#        drp_inc
#    ],
  link_with: [pgpdriver_lib],
)
