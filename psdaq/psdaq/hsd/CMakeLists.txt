find_package(PythonLibs REQUIRED)
find_package(nlohmann_json REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(hsd SHARED
  AdcCore.cc
  AdcSync.cc
  Adt7411.cc
  ClkSynth.cc
  #DmaCore.cc
  #FlashController.cc
  FexCfg.cc
  FmcCore.cc
  FmcSpi.cc
  Histogram.cc
  I2cSwitch.cc
  Jesd204b.cc
  LocalCpld.cc
  Mmcm.cc
  Pgp2b.cc
  Pgp3.cc
  PhyCore.cc
  PvDef.cc
  QABase.cc
  Tps2481.cc
  #Validator.cc

  Fmc134Ctrl.cc
  Fmc134Cpld.cc
  Ad7291.cc
  OptFmc.cc

  Module134.cc
  Module126.cc
  ModuleBase.cc
  ChipAdcReg.cc
)

target_include_directories(hsd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:include>
    ${PYTHON_INCLUDE_DIRS}
    ${Readline_INCLUDE_DIR}
)

target_link_libraries(hsd
    ${PYTHON_LIBRARIES}
    xtc
    service
    mmhw
    ${Readline_LIBRARY}
    rt
    nlohmann_json::nlohmann_json
)

add_executable(mempeek mempeek.cc)

target_link_libraries(mempeek
   mmhw
   dl
   rt
)

add_executable(hsd_eyescan hsd_eyescan.cc)

target_link_libraries(hsd_eyescan
   dl
   Threads::Threads
   service
   mmhw
   rt
)

add_executable(hsd134PVs hsd134PVs.cc PV134Ctrls.cc PVCtrlsBase.cc PV134Stats.cc)

target_link_libraries(hsd134PVs
    service
    epicstools
    ${EPICS_CA_LIB}
    ${EPICS_PVDATA_LIB}
    ${EPICS_PVACCESS_LIB}
    ${EPICS_PVACLIENT_LIB}
    Threads::Threads
    app
    rogue-core
    hsd
    mmhw
    rt
)

add_executable(hsd126PVs hsd126PVs.cc PV126Ctrls.cc PVCtrlsBase.cc PV126Stats.cc)

target_link_libraries(hsd126PVs
    service
    epicstools
    Threads::Threads
    app
    hsd
    mmhw
    rt
)


add_executable(hsd_sim hsd_sim.cc)
target_link_libraries(hsd_sim
    hsd
)

#add_executable(hsd_validate hsd_validate.cc)

#target_link_libraries(hsd_validate
#    Threads::Threads
#    hsd
#    mmhw
#    rt
#)

#add_executable(hsd_validate_sim hsd_validate_sim.cc)

#target_link_libraries(hsd_validate_sim
#    hsd
#)


add_executable(hsdRead hsdRead.cc ../../../psalg/psalg/digitizer/src/Stream.cc)

target_include_directories(hsdRead PUBLIC
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
     $<INSTALL_INTERFACE:include>
)

target_link_libraries(hsdRead
   hsd
   mmhw
   xtc
   epicstools
   Threads::Threads
   rt
)

install(TARGETS hsd
 		hsd126PVs
 		hsd134PVs
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
