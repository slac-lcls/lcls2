find_package(PythonLibs REQUIRED)

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
)

#---

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

#---

install(TARGETS
  trigger
  tmoTrigger
  tmoTeb
)
