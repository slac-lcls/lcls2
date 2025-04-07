find_package(PythonLibs REQUIRED)
find_package(nlohmann_json REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(trigger SHARED
  utilities.cc
)

target_include_directories(trigger PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  ${PYTHON_INCLUDE_DIRS}
)

target_link_libraries(trigger
  xtcdata::xtc
  service
  ${PYTHON_LIBRARIES}
  Threads::Threads
  nlohmann_json::nlohmann_json
)

#---

add_library(tmoTrigger SHARED
  TriggerPrimitiveExample_cam.cc
  TriggerPrimitiveExample_xpphsd.cc
  TriggerPrimitiveExample_bld.cc
  TriggerExample.cc
)

target_include_directories(tmoTrigger PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  ${PYTHON_INCLUDE_DIRS}
)

target_link_libraries(tmoTrigger
  xtcdata::xtc
  trigger
  nlohmann_json::nlohmann_json
)

#---

if(NOT CMAKE_CUDA_COMPILER)

  add_library(tmoTeb SHARED
    tmoTebPrimitive.cc
    tmoTeb.cc
  )

  target_include_directories(tmoTeb PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${PYTHON_INCLUDE_DIRS}
  )

  target_link_libraries(tmoTeb
    xtcdata::xtc
    trigger
  )

else()
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  set(CMAKE_CUDA_ARCHITECTURES 86 90)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
  endif()

  add_library(tmoTeb SHARED
    tmoTebPrimitive.cc
    tmoTebPrimitive_gpu.cu
    tmoTeb.cc
  )

  target_include_directories(tmoTeb PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${PYTHON_INCLUDE_DIRS}
  )

  target_link_libraries(tmoTeb
    xtcdata::xtc
    trigger
    CUDA::cudart
    CUDA::cuda_driver
  )

endif()

#---

add_library(tstTebPy SHARED
  timingTebPrimitive.cc
  tmoTebPrimitive.cc
  tebPyTrigger.cc
)

target_include_directories(tstTebPy PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  ${PYTHON_INCLUDE_DIRS}
)

target_link_libraries(tstTebPy
  xtcdata::xtc
  trigger
)

#---

add_library(calibTrigger SHARED
  calibTriggerDrp.cc
  calibTriggerTeb.cc
)

target_include_directories(calibTrigger PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  ${PYTHON_INCLUDE_DIRS}
)

target_link_libraries(calibTrigger
  xtcdata::xtc
  trigger
)

#---

install(FILES
  TimingTebData.hh
  TmoTebData.hh
  DESTINATION include/psdaq/trigger
)

install(TARGETS
  trigger
  tmoTrigger
  tmoTeb
  tstTebPy
  calibTrigger
)
