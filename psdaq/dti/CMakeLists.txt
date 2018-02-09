
add_library(dti SHARED
    PVStats.cc
    PVCtrls.cc
    Module.cc
)

target_include_directories(dti PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/epics/include
    $ENV{CONDA_PREFIX}/epics/include/os/Linux
)

find_library(YAML_CPP_LIB yaml-cpp $ENV{CONDA_PREFIX}/lib)
find_library(EPICS_CA_LIB ca $ENV{CONDA_PREFIX}/lib)
find_library(CPSW_LIB cpsw $ENV{CONDA_PREFIX}/lib)

#
#  dti_simple
#
add_executable(dti_simple dti_simple.cc)

target_link_libraries(dti_simple
   service
   cphw
   ${CPSW_LIB}
   ${YAML_CPP_LIB}
   dl
   pthread
   rt
)

#
#  dti_stats
#
add_executable(dti_stats dti_stats.cc)

target_link_libraries(dti_stats
   service
   cphw
   ${CPSW_LIB}
   ${YAML_CPP_LIB}
   dl
   pthread
   rt
)

#
#  dtiPVs
#
add_executable(dtiPVs dtiPVs.cc)

target_link_libraries(dtiPVs
   service
   cphw
   dti
   epicstools
   ${EPICS_CA_LIB}
   ${CPSW_LIB}
   ${YAML_CPP_LIB}
   dl
   pthread
   rt
)

#
#  dti_eyescan
#
add_executable(dti_eyescan dti_eyescan.cc)

target_link_libraries(dti_eyescan
   service
   cphw
   ${CPSW_LIB}
   ${YAML_CPP_LIB}
   dl
   pthread
   rt
)

#
#  dtiHsrCfg
#
add_executable(dtiHsrCfg dtiHsrCfg.cc)

target_link_libraries(dtiHsrCfg
   service
   cphw
   dti
   ${EPICS_CA_LIB}
   epicstools
   ${CPSW_LIB}
   ${YAML_CPP_LIB}
   dl
   pthread
   rt
)
