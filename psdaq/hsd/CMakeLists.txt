
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
  Pgp2bAxi.cc
  PhyCore.cc
  QABase.cc
  TprCore.cc
  Tps2481.cc
)

target_include_directories(hsd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    mmhw
)

#
#  hsd_eyescan
#
add_executable(hsd_eyescan hsd_eyescan.cc)

target_link_libraries(hsd_eyescan
   dl
   pthread
   service
   mmhw
)

#
#  hsdRead
#
add_executable(hsdRead hsdRead.cc)

target_include_directories(hsdRead PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(hsdRead
   pthread
   rt
)

#
#  hsd_init
#
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

#
#  hsd_pgp
#
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

#
#  hsd_sim
#
add_executable(hsd_sim hsd_sim.cc)

#
#  hsd_test
#
add_executable(hsd_test hsd_test.cc)

target_link_libraries(hsd_test
    pthread
    hsd
    mmhw 
    rt
)

#
# hsd_valid
#
add_executable(hsd_valid hsd_valid.cc)
