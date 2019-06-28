
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
  Jesd204b.cc
  LocalCpld.cc
  Mmcm.cc
  Module.cc
  Module64.cc
  Pgp2b.cc
  Pgp3.cc
  PhyCore.cc
  QABase.cc
  TprCore.cc
  Tps2481.cc
  Validator.cc

  Hsd3200.cc
  Fmc134Ctrl.cc
  Fmc134Cpld.cc
  Ad7291.cc
)

target_include_directories(hsd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(hsd
    xtcdata::xtc
    service
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
    $<INSTALL_INTERFACE:include>
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

add_executable(hsd_3200_i2c hsd_3200_i2c.cc)

target_include_directories(hsd_3200_i2c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(hsd_3200_i2c
    pthread
    hsd
    mmhw
    rt
)

add_executable(hsd_3200_reg hsd_3200_reg.cc)

target_link_libraries(hsd_3200_reg
    hsd
)
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
    pthread
    hsd
    mmhw
    rt
)

add_executable(hsd64PVs hsd64PVs.cc PV64Ctrls.cc PV64Stats.cc)

target_link_libraries(hsd64PVs
    service
    epicstools
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

target_include_directories(hsd_validate PUBLIC
    ../pgp/aes-stream-drivers/include
)

target_link_libraries(hsd_validate
    pthread
    hsd
    mmhw
    rt
)

add_executable(hsd_validate_sim hsd_validate_sim.cc)

target_link_libraries(hsd_validate_sim
    hsd
)


add_executable(hsdRead hsdRead.cc ../../../psalg/psalg/digitizer/src/Stream.cc)

target_include_directories(hsdRead PUBLIC
     $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
     $<INSTALL_INTERFACE:include>
)

target_link_libraries(hsdRead
   hsd
   mmhw
   xtcdata::xtc
   epicstools
   pthread
   rt
)

add_executable(hsd_promload promload.cc)
target_include_directories(hsdRead PUBLIC
     $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(hsd_promload
   hsd
   mmhw
   rt
)


install(TARGETS hsd
                hsd_promload
		hsd_eyescan
		hsd_init
		hsd_phase
		hsd_pgp
		hsd_test
 		hsdRead
 		hsdPVs
 		hsd64PVs
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
