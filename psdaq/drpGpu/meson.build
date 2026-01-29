

if cuda_compiler.found()
  cuda_args = [
    '-gencode=arch=compute_86,code=sm_86',
    '-gencode=arch=compute_90,code=sm_90',
    '-Wno-deprecated-gpu-targets',
    #'-G', #for debugging
    #'--verbose',
  ]

  cuda_dep = declare_dependency(
    include_directories: [
      include_directories(join_paths(conda_base, 'targets/x86_64-linux/include/')),
      include_directories(join_paths(conda_base, 'targets/x86_64-linux/include/cccl/')),
    ],
    link_args: [
      '-L' + join_paths(join_paths(conda_base, 'targets/x86_64-linux/lib/')),
      '-lcudart',
      '-lcuda',
   ]
  )

  drpgpu_inc_dir = include_directories('..')

  cufile_dep = cc.find_library('cufile', dirs: join_paths(conda_base, 'lib'))

  drpbase_gpu_lib = shared_library('drpbase_gpu',
    [
      'MemPool.cc',
      'GpuAsyncLib.cc',
    ],
    include_directories: [
      drpgpu_inc_dir,
      include_directories('../../psalg/'),
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [
      drpbase_lib,
      collection_lib,
    ],
    dependencies: [
      cuda_dep,
      psalg_utils_dep,
    ],
  )

  executable('drp_gpu',
    [
      'PGPDetectorApp.cc',
      'PGPDetector.cc',
      'Reader.cu',
      'Collector.cu',
      'Reducer.cu',
      'FileWriter.cc',
    ],
    include_directories: [
      drpgpu_inc_dir,
    ],
    dependencies: [
      cuda_dep,
      xtcdata_dep,
      python_dep,
      psalg_utils_dep,
      epics_dep,
      zmq_dep,
      prometheus_pull_dep,
      cufile_dep,
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [
      drpbase_gpu_lib,
      epicstools_lib,
      collection_lib,
      drpbase_lib,
      service_lib,
      exporter_lib,
      contributor_lib,
      utilities_lib,
    ],
    #pie: true,
    cuda_args: cuda_args,
  )

  AreaDetector_gpu_lib = shared_library('AreaDetector_gpu',
    [
      'Detector.cc',
      'AreaDetector.cu',
      '../drp/AreaDetector.cc',
    ],
    include_directories: [
      drpgpu_inc_dir,
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [],
    dependencies: [
      python_dep,
      cuda_dep,
      xtcdata_dep,
      psalg_utils_dep,
    ],
  )

  EpixUHRemu_gpu_lib = shared_library('EpixUHRemu_gpu',
    [
      'Detector.cc',
      'EpixUHRemu.cu',
      '../drp/AreaDetector.cc',
    ],
    include_directories: [drpgpu_inc_dir],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [],
    dependencies: [
      xtcdata_dep,
      psalg_utils_dep,
      cuda_dep,
      python_dep,
    ],
  )

  NoOpReducer_lib = shared_library('NoOpReducer',
    'NoOpReducer.cu',
    include_directories: [drpgpu_inc_dir],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [],
    dependencies: [
      #python_dep,
      xtcdata_dep,
      psalg_utils_dep,
    ],
  )

  #https://github.com/burtscher/LC-framework
  #LcReducer_lib = shared_library('LcReducer',
  #  'LcReducer.cu',
  #  include_directories: [
  #    drpgpu_inc_dir,
  #    lc_inc_dir,
  #  ],
  #  install: true,
  #  install_rpath: libdir_install_rpath,
  #  link_with: [],
  #  dependencies: [
  #    xtcdata_dep,
  #    cuda_dep,
  #    #lc_dep,
  #  ],
  #)

  #https://github.com/burtscher/PFPL
  #PfplReducer_lib = shared_library('PfplReducer',
  #  'PfplReducer.cu',
  #  include_directories: [drpgpu_inc_dir],
  #  install: true,
  #  install_rpath: libdir_install_rpath,
  #  link_with: [],
  #  dependencies: [
  #    xtcdata_dep,
  #  ],
  #)

  #https://github.com/szcompressor/cuSZ
  #CuSzReducer_lib = shared_library('CuSzReducer',
  #  'CuSzReducer.cu',
  #  include_directories: [drpgpu_inc_dir],
  #  install: true,
  #  install_rpath: libdir_install_rpath,
  #  link_with: [],
  #  dependencies: [
  #    #xtcdata_dep,
  #  ],
  #)

  #https://github.com/szcompressor/cuSZp
  #CuSZpReducer_lib = shared_library('CuSZpReducer',
  #  'CuSZpReducer.cu',
  #  include_directories: [drpgpu_inc_dir],
  #  install: true,
  #  install_rpath: libdir_install_rpath,
  #  link_with: [],
  #  dependencies: [
  #    xtcdata_dep,
  #  ],
  #)

  executable('pgpread_gpu',
    [
      'pgpread.cc',
      'GpuAsyncLib.cc',
    ],
    include_directories: [drpgpu_inc_dir],
    dependencies: [
      cuda_dep,
      psalg_utils_dep,
      threads_dep,
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [],
    #pie: true,
    cuda_args: cuda_args,
  )

  executable('cudaDeviceProperties',
    [
      'cudaDeviceProperties.cu',
    ],
    include_directories: [drpgpu_inc_dir],
    dependencies: [],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [],
    #pie: true,
    cuda_args: cuda_args,
  )

  executable('calibTest',
    [
      'calibTest.cu',
      'GpuAsyncLib.cc',
    ],
    include_directories: [drpgpu_inc_dir],
    dependencies: [
      psalg_utils_dep,
      cuda_dep,
    ],
    #install: true,
    #install_rpath: libdir_install_rpath,
    link_with: [],
    #pie: true,
    cuda_args: cuda_args,
  )
endif
