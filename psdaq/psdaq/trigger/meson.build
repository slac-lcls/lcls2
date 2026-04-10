
trigger_inc_dir = include_directories('../..')

trigger_lib = shared_library('trigger',
  'src/utilities.cc',
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
    'src/mfxTripperPrimitive.cc',
    'src/mfxTripperPrimitive_jungfrau.cc',
    'src/mfxTripperTeb.cc',
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
    'src/TriggerPrimitiveExample_cam.cc',
    'src/TriggerPrimitiveExample_xpphsd.cc',
    'src/TriggerPrimitiveExample_bld.cc',
    'src/TriggerExample.cc',
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
    'src/tmoTebPrimitive.cc',
    'src/tmoTeb.cc',
  ],
  include_directories: [trigger_inc_dir],
  dependencies: [
    xtcdata_dep,
    psalg_utils_dep,
  ],
  install: true,
  install_rpath: libdir_install_rpath,
)
cuda_compiler = find_program('nvcc', required: false)
if cuda_compiler.found()
  cuda_args = [
    '-gencode=arch=compute_86,code=sm_86',
    '-gencode=arch=compute_90,code=sm_90',
    '-Wno-deprecated-gpu-targets',
    '-Xcompiler=-fPIC',
    #'-G', #for debugging
    #'--verbose',
  ]
  cuda_inc = include_directories('../..')
  tmoTeb_gpu_lib = shared_library('tmoTeb_gpu',
    [
      'src/tmoTebPrimitive.cc',
      'src/tmoTebPrimitive_gpu.cu',
      'src/tmoTeb.cc'
    ],
    include_directories: [trigger_inc_dir],
    dependencies: [
      psalg_utils_dep,
      xtcdata_dep,
      nlohmann_json_dep,
    ],
    #link_with: [],
    install: true,
    install_rpath: libdir_install_rpath,
    cuda_args: cuda_args,
  )
endif

tstTebPy_lib = shared_library('tstTebPy',
  [
    'src/timingTebPrimitive.cc',
    'src/tmoTebPrimitive.cc',
    'src/tebPyTrigger.cc',
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
    'src/calibTriggerDrp.cc',
    'src/calibTriggerTeb.cc',
  ],
  include_directories: [trigger_inc_dir],
  dependencies: [
    xtcdata_dep,
    psalg_utils_dep,
  ],
  install: true,
  install_rpath: libdir_install_rpath,
)

#install_headers(
#  [
#    'TimingTebData.hh',
#    'TmoTebData.hh',
#  ],
#  subdir: 'psdaq/trigger'
#)
