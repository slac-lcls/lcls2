
add_library(hsd SHARED
  AdcCore.cc
  AdcSync.cc
  Adt7411.cc
  ClkSynth.cc
  DmaCore.cc
  FlashController.cc
  FmcCore.cc
  FmcSpi.cc
  Histogram.cc
  I2cSwitch.cc
  LocalCpld.cc
  Mmcm.cc
  Module.cc
  Pgp2b.cc
  Pgp3.cc
  PhyCore.cc
  QABase.cc
  TprCore.cc
  Tps2481.cc

  Hsd3200.cc
  Fmc134Ctrl.cc
  Fmc134Cpld.cc
  Ad7291.cc
)

target_include_directories(hsd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    mmhw
)

target_link_libraries(hsd
    xtcdata::xtc
    mmhw
    rt
)

add_executable(mempeek mempeek.cc)

target_link_libraries(mempeek
   dl
   rt
)

add_executable(hsd_eyescan hsd_eyescan.cc)

target_link_libraries(hsd_eyescan
   dl
   pthread
   service
   mmhw
   rt
)

add_executable(hsd_init hsd_init.cc)

target_include_directories(hsd_init PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(hsd_init
    pthread
    hsd
    mmhw
    rt
)

add_executable(hsd_3200 hsd_3200.cc)

target_include_directories(hsd_3200 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(hsd_3200
    pthread
    hsd
    mmhw
    rt
)

add_executable(hsd_3200_reg hsd_3200_reg.cc)

target_include_directories(hsd_3200_reg PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(hsd_3200_reg
    pthread
    rt
)

add_executable(hsd_phase hsd_phase.cc)

target_include_directories(hsd_phase PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(hsd_phase
    pthread
    hsd
    mmhw
    rt
)

add_executable(hsd_clocktree hsd_clocktree.cc)

target_include_directories(hsd_clocktree PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(hsd_clocktree
    pthread
    hsd
    mmhw
    rt
)

add_executable(hsd_pgp hsd_pgp.cc)

target_include_directories(hsd_pgp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(hsd_pgp
    pthread
    hsd
    mmhw
    rt
)

add_executable(hsdPVs hsdPVs.cc PVCtrls.cc PVStats.cc)

target_include_directories(hsd_pgp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(hsdPVs
    service
    epicstools
    ${EPICS_CA_LIB}
    ${EPICS_PVDATA_LIB}
    ${EPICS_PVACCESS_LIB}
    ${EPICS_PVACLIENT_LIB}
    pthread
    hsd
    mmhw
    rt
)

add_executable(hsd_sim hsd_sim.cc)
target_link_libraries(hsd_sim
    hsd
)

add_executable(hsd_test hsd_test.cc)

target_link_libraries(hsd_test
    pthread
    hsd
    mmhw
    rt
)


add_executable(hsd_flash hsd_flash.cc)

target_include_directories(hsd_flash PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)
target_link_libraries(hsd_flash
    pthread
    hsd
    mmhw
    rt
)


add_executable(hsd_validate hsd_validate.cc)

target_link_libraries(hsd_validate
    pthread
    hsd
    mmhw
    rt
)


add_executable(hsdRead hsdRead.cc ../../../psalg/psalg/digitizer/src/Stream.cc)

target_include_directories(hsdRead PUBLIC
     $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
     $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/../install/include>
     $<INSTALL_INTERFACE:../install/include>
)

target_link_libraries(hsdRead
   hsd
   xtcdata::xtc
   epicstools
   ${EPICS_CA_LIB}
   ${EPICS_PVDATA_LIB}
   ${EPICS_PVACCESS_LIB}
   ${EPICS_PVACLIENT_LIB}
   pthread
   rt
)

install(TARGETS hsd
		hsd_eyescan
		hsd_init
		hsd_phase
		hsd_pgp
		hsd_test
 		hsdRead
 		hsdPVs
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
