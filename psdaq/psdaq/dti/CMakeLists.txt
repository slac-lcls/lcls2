
add_library(dti SHARED
    PVStats.cc
    PVCtrls.cc
    Module.cc
)

target_include_directories(dti PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(dti
    epicstools
    cpsw::cpsw
    ${YAML_CPP_LIB}
)

#
#  dti_simple
#
add_executable(dti_simple dti_simple.cc)

target_link_libraries(dti_simple
   service
   cphw
   cpsw::cpsw
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
   cpsw::cpsw
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
   dl
   pthread
   rt
)

#
#  dtiPVs
#
add_executable(dtiPVr dtiPVs.cc)

target_link_libraries(dtiPVr
   service
   cphwr
   dti
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
   cpsw::cpsw
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
   dl
   pthread
   rt
)

#
#  dti_appreg
#
add_executable(dti_appreg dti_appreg.cc)

target_link_libraries(dti_appreg
   service
   cphw
   cpsw::cpsw
   ${YAML_CPP_LIB}
   dl
   pthread
   rt
)


install(TARGETS dti
		dti_simple
		dti_stats
		dtiPVs
		dti_eyescan
		dti_appreg
		dtiHsrCfg
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
