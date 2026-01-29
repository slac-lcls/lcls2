
trigger_inc_dir = include_directories('../..')

trigger_lib = shared_library('trigger',
  'utilities.cc',
  include_directories: [trigger_inc_dir],
  dependencies: [
    psalg_utils_dep,
    python_dep,
  ],
  install: true,
  install_rpath: libdir_install_rpath,
)

mfxTripperTeb_lib = shared_library('mfxTripperTeb',
  [
    'mfxTripperPrimitive.cc',
    'mfxTripperPrimitive_jungfrau.cc',
    'mfxTripperTeb.cc',
  ],
  include_directories: [
    trigger_inc_dir,
    include_directories(join_paths(get_option('epics_base'), 'include')),
  ],
  dependencies: [
    xtcdata_dep,
    psalg_utils_dep,
    epics_dep,
  ],
  install: true,
  install_rpath: libdir_install_rpath,
)

tmoTrigger_lib = shared_library('tmoTrigger',
  [
    'TriggerPrimitiveExample_cam.cc',
    'TriggerPrimitiveExample_xpphsd.cc',
    'TriggerPrimitiveExample_bld.cc',
    'TriggerExample.cc',
  ],
  include_directories: [trigger_inc_dir],
  dependencies: [
    xtcdata_dep,
    psalg_utils_dep,
  ],
  install: true,
  install_rpath: libdir_install_rpath,
)

tmoTeb_lib = shared_library('tmoTeb',
  [
    'tmoTebPrimitive.cc',
    'tmoTeb.cc',
  ],
  include_directories: [trigger_inc_dir],
  dependencies: [
    xtcdata_dep,
    psalg_utils_dep,
  ],
  install: true,
  install_rpath: libdir_install_rpath,
)

if cuda_compiler.found()
  cuda_args = [
    '-gencode=arch=compute_86,code=sm_86',
    '-gencode=arch=compute_90,code=sm_90',
    '-Wno-deprecated-gpu-targets',
    '--compiler-options=-fPIC',
    #'-G', #for debugging
    #'--verbose',
  ]
  cuda_inc = include_directories('../..')
  tmoTeb_gpu_lib = shared_library('tmoTeb_gpu',
    [
      'tmoTebPrimitive.cc',
      'tmoTebPrimitive_gpu.cu',
      'tmoTeb.cc'
    ],
    include_directories: [trigger_inc_dir],
    dependencies: [psalg_utils_dep, xtcdata_dep],
    #link_with: [],
    install: true,
    install_rpath: libdir_install_rpath,
  )
endif

tstTebPy_lib = shared_library('tstTebPy',
  [
    'timingTebPrimitive.cc',
    'tmoTebPrimitive.cc',
    'tebPyTrigger.cc',
  ],
  include_directories: [trigger_inc_dir],
  dependencies: [
    xtcdata_dep,
    psalg_utils_dep,
  ],
  link_with: [
    service_lib,
  ],
  install: true,
  install_rpath: libdir_install_rpath,
)

calibTrigger_lib = shared_library('calibTrigger',
  [
    'calibTriggerDrp.cc',
    'calibTriggerTeb.cc',
  ],
  include_directories: [trigger_inc_dir],
  dependencies: [
    xtcdata_dep,
    psalg_utils_dep,
  ],
  install: true,
  install_rpath: libdir_install_rpath,
)

install_headers(
  [
    'TimingTebData.hh',
    'TmoTebData.hh',
  ],
  subdir: 'psdaq/trigger'
)
