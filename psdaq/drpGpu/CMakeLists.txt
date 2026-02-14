find_package(PythonLibs REQUIRED)
find_package(nlohmann_json REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
    endif()

    #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --verbose")  # or do: export NVCC_PREPEND_FLAGS='--verbose'
    #message(CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")
    #message(INSTALL_INTERFACE="$<INSTALL_INTERFACE>")
    #message(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES="${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    #message(CMAKE_CXX_COMPILER='${CMAKE_CXX_COMPILER}')
    #message(NVCC_PREPEND_FLAGS='$ENV{NVCC_PREPEND_FLAGS}')

#--- DRP program

    add_library(drpbase_gpu SHARED
        MemPool.cc
        GpuAsyncLib.cc
    )
    target_include_directories(drpbase_gpu PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(drpbase_gpu
        drpbase
        CCCL::CCCL              # Needed with CUDA 13 to pick up <cuda/atomic>
    )

    add_executable(drp_gpu
        PGPDetectorApp.cc
        PGPDetector.cc
        Reader.cu
        Collector.cu
        Reducer.cu
        FileWriter.cc
    )
    target_include_directories(drp_gpu PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(drp_gpu
        drpbase_gpu
        trigger
        mmhw
        xtcdata::xtc
        nlohmann_json::nlohmann_json
        #cudadevrt
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cuFile            # cufile is not found, but cuFile is
    )
    #target_compile_options(drp_gpu PUBLIC -O3)
    #target_link_options(drp_gpu PUBLIC "-rdc=true")

#--- Detector libraries

    add_library(AreaDetector_gpu SHARED
        Detector.cc
        AreaDetector.cu
        ../drp/AreaDetector.cc  # @todo: Better solution than '../drp/'?
    )
    target_include_directories(AreaDetector_gpu PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(AreaDetector_gpu
        drpbase_gpu
        xtcdata::xtc
        nlohmann_json::nlohmann_json
        CUDA::cudart
        CUDA::cuda_driver
    )
    #target_compile_options(AreaDetector_gpu PUBLIC -O3)

    add_library(EpixUHRemu_gpu SHARED
        Detector.cc
        EpixUHRemu.cu
        ../drp/AreaDetector.cc  # @todo: Better solution than '../drp/'?
    )
    target_include_directories(EpixUHRemu_gpu PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(EpixUHRemu_gpu
        drpbase_gpu
        xtcdata::xtc
        nlohmann_json::nlohmann_json
        CUDA::cudart
        CUDA::cuda_driver
    )
    #target_compile_options(EpixUHRemu_gpu PUBLIC -O3)

    add_library(EpixUHRsim_gpu SHARED
        Detector.cc
        SimDetector.cc
        EpixUHRsim.cu
        ../drp/AreaDetector.cc  # @todo: Better solution than '../drp/'?
    )
    target_include_directories(EpixUHRsim_gpu PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(EpixUHRsim_gpu
        drpbase_gpu
        xtcdata::xtc
        nlohmann_json::nlohmann_json
        CUDA::cudart
        CUDA::cuda_driver
    )
    #target_compile_options(EpixUHRsim_gpu PUBLIC -O3)

#--- Reducer libraries

    # Add the following to your environment instead of appending to the hardcoded CMAKE_PREFIX_PATH:
    #   export LC_DIR=~/git/psdaq-reducers/install/cmake
    #   export PFPL_DIR=~/git/psdaq-reducers/install/cmake
    #   export CUSZ_DIR=~/git/psdaq-reducers/install/lib64/cmake/CUSZ
    #   export cuSZp_DIR=~/git/psdaq-reducers/install/cmake

    add_library(NoOpReducer SHARED
        NoOpReducer.cu
    )
    target_include_directories(NoOpReducer PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(NoOpReducer
        xtcdata::xtc
        nlohmann_json::nlohmann_json
        CUDA::cudart
        CUDA::cuda_driver
    )
    #target_compile_options(NoOpReducer PUBLIC -O3)

#--- LC

    find_package(LC REQUIRED)

    add_library(LcReducer SHARED
        LcReducer.cu
    )
    target_include_directories(LcReducer PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(LcReducer
        LC::lc_shared
        xtcdata::xtc
        nlohmann_json::nlohmann_json
        CUDA::cudart
        CUDA::cuda_driver
    )
    #target_compile_options(LcReducer PUBLIC -O3)

#--- PFPL

    find_package(PFPL REQUIRED)

    add_library(PfplReducer SHARED
        PfplReducer.cu
    )
    target_include_directories(PfplReducer PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(PfplReducer
        PFPL::pfpl_shared
        xtcdata::xtc
        nlohmann_json::nlohmann_json
        CUDA::cudart
        CUDA::cuda_driver
    )
    #target_compile_options(PfplReducer PUBLIC -O3)

#--- cuSZ

    find_package(CUSZ REQUIRED)

    add_library(CuSzReducer SHARED
        CuSzReducer.cu
    )
    target_include_directories(CuSzReducer PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(CuSzReducer
        CUSZ::cusz
        xtcdata::xtc
        nlohmann_json::nlohmann_json
        CUDA::cudart
        CUDA::cuda_driver
    )
    #target_compile_options(CuSzReducer PUBLIC -O3)

#--- cuSZp

    find_package(cuSZp REQUIRED)

    add_library(CuSZpReducer SHARED
        CuSZpReducer.cu
    )
    target_include_directories(CuSZpReducer PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(CuSZpReducer
        cuSZp::cuSZp_shared
        xtcdata::xtc
        nlohmann_json::nlohmann_json
        CUDA::cudart
        CUDA::cuda_driver
    )
    #target_compile_options(CuSZpReducer PUBLIC -O3)

#--- Tools

    add_executable(pgpread_gpu
        pgpread.cc
        GpuAsyncLib.cc
    )
    target_include_directories(pgpread_gpu PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        ${CMAKE_INSTALL_PREFIX}/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(pgpread_gpu
        CUDA::cudart
        CUDA::cuda_driver
        Threads::Threads
    )

    add_executable(cudaDeviceProperties
        cudaDeviceProperties.cu
    )
    target_include_directories(cudaDeviceProperties PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        ${CMAKE_INSTALL_PREFIX}/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(cudaDeviceProperties
        CUDA::cudart
        CUDA::cuda_driver
    )

#--- Tests

    add_executable(calibTest
        calibTest.cu
        GpuAsyncLib.cc
    )
    target_include_directories(calibTest PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        ${CMAKE_INSTALL_PREFIX}/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(calibTest
        CUDA::cudart
        CUDA::cuda_driver
        Threads::Threads
    )

    add_executable(fileWriteTest_gpu
        fileWriteTest.cc
        GpuAsyncLib.cc
        FileWriter.cc
    )
    target_include_directories(fileWriteTest_gpu PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        ${CMAKE_INSTALL_PREFIX}/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
  target_link_libraries(fileWriteTest_gpu
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cuFile            # cufile is not found, but cuFile is
        Threads::Threads
    )

#--- Install

    install(TARGETS
        drpbase_gpu
        drp_gpu
        AreaDetector_gpu
        EpixUHRemu_gpu
        EpixUHRsim_gpu
        NoOpReducer
        #LcReducer_old
        LcReducer
        PfplReducer
        CuSzReducer
        CuSZpReducer
        pgpread_gpu
        cudaDeviceProperties
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()
