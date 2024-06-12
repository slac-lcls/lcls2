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

add_library(tstTebPy SHARED
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
