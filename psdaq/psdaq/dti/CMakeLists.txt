set(THREADS_PREFER_PTHREAD_FLAG ON)
find_packages(Threads REQUIRED)

add_library(dti SHARED
    PVStats.cc
    PVCtrls.cc
    Module.cc
)

target_include_directories(dti PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
)

target_link_libraries(dti
    epicstools
    ${YAML_CPP_LIB}
)

#
#  dti_simple
#
add_executable(dti_simple dti_simple.cc)

target_link_libraries(dti_simple
   service
   cphw
   ${YAML_CPP_LIB}
   dl
   Threads::Threads
   rt
)

#
#  dti_stats
#
add_executable(dti_stats dti_stats.cc)

target_link_libraries(dti_stats
   service
   cphw
   ${YAML_CPP_LIB}
   dl
   Threads::Threads
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
   Threads::Threads
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
   Threads::Threads
   rt
)

#
#  dti_eyescan
#
add_executable(dti_eyescan dti_eyescan.cc)

target_link_libraries(dti_eyescan
   service
   cphw
   ${YAML_CPP_LIB}
   dl
   Threads::Threads
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
   Threads::Threads
   rt
)

#
#  dti_appreg
#
add_executable(dti_appreg dti_appreg.cc)

target_link_libraries(dti_appreg
   service
   cphw
   ${YAML_CPP_LIB}
   dl
   Threads::Threads
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
