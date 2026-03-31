
if cuda_compiler.found()
  cuda_args = [
    '-gencode=arch=compute_86,code=sm_86',
    '-gencode=arch=compute_90,code=sm_90',
    '-Wno-deprecated-gpu-targets',
    '-Xcompiler=-fPIC',
    #'-G', #for debugging
    #'--verbose',
  ]

  if cuda_base != ''
    # using CUDA_ROOT
    cuda_path = cuda_base
  else
    # use cuda in conda
    cuda_path = conda_base
  endif

  cuda_inc_dir = include_directories([
      join_paths(cuda_path, 'targets/x86_64-linux/include'),
      join_paths(cuda_path, 'targets/x86_64-linux/include/cccl/')
    ]
  )

  cuda_lib_dirs = [
    join_paths(cuda_path, 'targets/x86_64-linux/lib'),
    #join_paths(cuda_path, 'lib64/stubs'), # Using stubs from toolkit
    join_paths(cuda_path, 'lib/'),
    join_paths(cuda_path, 'lib64/'),
  ]

  cudart_dep = cc.find_library('cudart', dirs: cuda_lib_dirs)
  cuda_main_dep = cc.find_library('cuda',
    dirs: [
      cuda_lib_dirs,
      join_paths(cuda_path, 'lib64/stubs'), # Only needed in Python 3.9
    ]
  )
  cufile_dep = cc.find_library('cufile', dirs: cuda_lib_dirs)
  # Patched cufile lib with no glibc restriction for log2f
  #cufile_dep = cc.find_library('cufile', dirs: '/cds/home/k/kmecseki/cuda2/')

  cuda_dep = declare_dependency(
    include_directories: cuda_inc_dir,
    dependencies: [cudart_dep, cuda_main_dep],
  )

  subproj_options = ['conda_prefix=' + cuda_path]
  lc_proj = subproject('lc', default_options: subproj_options)
  lc_dep = lc_proj.get_variable('lc_dep')
  pfpl_proj = subproject('pfpl', default_options: subproj_options)
  pfpl_dep = pfpl_proj.get_variable('pfpl_dep')

  cusz_dep  = cusz_proj.dependency('psz_cu_compile_settings')
  cuszp_dep = cuszp_proj.dependency('cuSZp_shared')

  drpgpu_inc_dir = include_directories('..')

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

  # Remove link_args when using newer conda (with sysroot 2.28< (kernel 4.18<))
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
      nlohmann_json_dep,
      #threads_dep,
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
    link_args: [
      '/lib64/libcuda.so.1',
      '/lib64/librt.so.1',
      '/lib64/libpthread.so.0',
      '/lib64/libdl.so.2',
      '/lib64/libc.so.6',
      '/lib64/libm.so.6',
      '/lib64/ld-linux-x86-64.so.2',
      '/lib64/libutil.so.1',
      '/lib64/libresolv.so.2',
    ],
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
      nlohmann_json_dep,
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
      nlohmann_json_dep,
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
      nlohmann_json_dep,
    ],
  )

  LcReducer_lib = shared_library('LcReducer',
    'LcReducer.cu',
    include_directories: [
      drpgpu_inc_dir,
      #lc_inc_dir,
    ],
    install: true,
    install_rpath: libdir_install_rpath,
    dependencies: [
      xtcdata_dep,
      cuda_dep,
      lc_dep,
      psalg_utils_dep,
      nlohmann_json_dep,
    ],
  )

  PfplReducer_lib = shared_library('PfplReducer',
    'PfplReducer.cu',
    include_directories: [drpgpu_inc_dir],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [],
    dependencies: [
      xtcdata_dep,
      pfpl_dep,
      psalg_utils_dep,
      nlohmann_json_dep,
    ],
  )

  CuSzReducer_lib = shared_library('CuSzReducer',
    'CuSzReducer.cu',
    include_directories: [drpgpu_inc_dir],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [],
    dependencies: [
      xtcdata_dep,
      cusz_dep,
      psalg_utils_dep,
      nlohmann_json_dep,
    ],
  )

  CuSZpReducer_lib = shared_library('CuSZpReducer',
    'CuSZpReducer.cu',
    include_directories: [drpgpu_inc_dir],
    install: true,
    install_rpath: libdir_install_rpath,
    link_with: [],
    dependencies: [
      xtcdata_dep,
      cuszp_dep,
      psalg_utils_dep,
      nlohmann_json_dep,
    ],
  )

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
