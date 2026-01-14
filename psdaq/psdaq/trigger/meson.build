python_dep = dependency('python3')
#json_dep = dependency('nlohmann_json', required : true)
#threads_dep = dependency('threads')
#xtcdata_dep = dependency('xtcdata')   # adjust as needed
#service_dep = dependency('service')   # adjust as needed
#epics_ca_dep = dependency('epics::ca', required : false)  # optional

trigger_inc_dir = include_directories('../..')

trigger_lib = shared_library('trigger',
  'utilities.cc',
  include_directories: [trigger_inc_dir],
  dependencies: [
    psalg_utils_dep,
    python_dep
  ],
)

# ---------------------------
# mfxTripperTeb library
# ---------------------------
#mfxTripperTeb_lib = shared_library('mfxTripperTeb',
#  [
#    'mfxTripperPrimitive.cc',
#    'mfxTripperPrimitive_jungfrau.cc',
#    'mfxTripperTeb.cc'
#  ],
#  include_directories: [trigger_inc_dir],
#  dependencies: [xtcdata_dep, trigger_lib, json_dep, epics_ca_dep],
#  install: true
#)

# ---------------------------
# tmoTrigger library
# ---------------------------
#tmoTrigger_lib = shared_library('tmoTrigger',
#  [
#    'TriggerPrimitiveExample_cam.cc',
#    'TriggerPrimitiveExample_xpphsd.cc',
#    'TriggerPrimitiveExample_bld.cc',
#    'TriggerExample.cc'
#  ],
#  include_directories: [trigger_inc_dir],
#  dependencies: [xtcdata_dep, trigger_lib, json_dep],
#  install: true
#)

# ---------------------------
# tmoTeb library
# ---------------------------
#tmoTeb_lib = shared_library('tmoTeb',
#  [
#    'tmoTebPrimitive.cc',
#    'tmoTeb.cc'
#  ],
#  include_directories: [trigger_inc_dir],
#  dependencies: [xtcdata_dep, trigger_lib],
#  install: true
#)

# ---------------------------
# tmoTeb GPU library (if CUDA enabled)
# ---------------------------
#cuda_compiler = find_program('nvcc', required : false)

#if cuda_compiler.found()
#  cuda_inc = include_directories('../..')
#  tmoTeb_gpu_lib = shared_library('tmoTeb_gpu',
#    [
#      'tmoTebPrimitive.cc',
#      'tmoTebPrimitive_gpu.cu',
#      'tmoTeb.cc'
#    ],
#    include_directories: [trigger_inc_dir],
#    dependencies: [xtcdata_dep, trigger_lib],
#    link_with: [],   # CUDA runtime will be automatically linked
#    install: true,
#    # specify CUDA compilation flags
#    override_options: ['cpp_std=c++17']
#  )
#endif

# ---------------------------
# tstTebPy library
# ---------------------------
#tstTebPy_lib = shared_library('tstTebPy',
#  [
#    'timingTebPrimitive.cc',
#    'tmoTebPrimitive.cc',
#    'tebPyTrigger.cc'
#  ],
#  include_directories: [trigger_inc_dir],
#  dependencies: [xtcdata_dep, trigger_lib],
#  install: true
#)

# ---------------------------
# calibTrigger library
# ---------------------------
#calibTrigger_lib = shared_library('calibTrigger',
#  [
#    'calibTriggerDrp.cc',
#    'calibTriggerTeb.cc'
#  ],
#  include_directories: [trigger_inc_dir],
#  dependencies: [xtcdata_dep, trigger_lib],
#  install: true
#)

# ---------------------------
# headers installation
# ---------------------------
#install_headers(['TimingTebData.hh', 'TmoTebData.hh'],
#  subdir: 'psdaq/trigger'
#)

# ---------------------------
# libraries installation
# ---------------------------
#install_targets = [
#  trigger_lib,
#  mfxTripperTeb_lib,
#  tmoTrigger_lib,
#  tmoTeb_lib,
#  tstTebPy_lib,
#  calibTrigger_lib
#]

#if cuda_compiler.found()
#  install_targets += [tmoTeb_gpu_lib]
#endif

#install_targets(install_targets)