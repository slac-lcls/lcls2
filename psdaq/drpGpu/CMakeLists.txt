find_package(PythonLibs REQUIRED)
find_package(nlohmann_json REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    set(CMAKE_CUDA_ARCHITECTURES 86 90)
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
        MemPool.cu
        GpuAsyncLib.cc
    )
    target_include_directories(drpbase_gpu PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(drpbase_gpu
        drpbase
    )

    add_executable(drp_gpu
        PGPDetectorApp.cc
        PGPDetector.cu
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
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cuFile            # cufile is not found, but cuFile is
    )
    set_target_properties(drp_gpu PROPERTIES CUDA_ARCHITECTURES "86")
    #target_compile_options(drp_gpu PUBLIC -O3)

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
    set_target_properties(AreaDetector_gpu PROPERTIES CUDA_ARCHITECTURES "86")
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
    set_target_properties(EpixUHRemu_gpu PROPERTIES CUDA_ARCHITECTURES "86")
    #target_compile_options(EpixUHRemu_gpu PUBLIC -O3)

#--- Reducer libraries

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
    set_target_properties(NoOpReducer PROPERTIES CUDA_ARCHITECTURES "86")
    #target_compile_options(NoOpReducer PUBLIC -O3)

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

    install(TARGETS
        drpbase_gpu
        drp_gpu
        AreaDetector_gpu
        EpixUHRemu_gpu
        NoOpReducer
        pgpread_gpu
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()
