
add_library(dti SHARED
    PVStats.cc
    PVCtrls.cc
    Module.cc
)

target_include_directories(dti PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    /reg/common/package/epicsca/3.14.12/include
    /reg/common/package/epicsca/3.14.12/include/os/Linux
)

find_library(YAML_CPP_LIBRARY yaml-cpp HINTS $ENV{CONDA_PREFIX}/lib)
find_library(CPSW_LIBRARY cpsw HINTS $ENV{CONDA_PREFIX}/lib)

#
#  dti_simple
#
add_executable(dti_simple dti_simple.cc)

target_link_libraries(dti_simple
   service
   cphw
   #cpsw
   ${CPSW_LIBRARY}
   ${YAML_CPP_LIBRARY}
   dl
   pthread
   rt
   #yaml-cpp
   #/reg/common/package/cpsw/framework-R3-4/x86_64-rhel7-opt/lib/libcpsw.so
   #/reg/common/package/yaml-cpp/yaml-cpp-0.5.3/x86_64-rhel7-gcc48-opt/lib/libyaml-cpp.so
)

#
#  dtiPVs
#
add_executable(dtiPVs dtiPVs.cc)

target_link_libraries(dtiPVs
   service
   cphw
   dti
   /reg/common/package/epicsca/3.14.12/lib/x86_64-rhel7-opt/libca.so
   epicstools
   ${CPSW_LIBRARY}
   ${YAML_CPP_LIBRARY}
   #cpsw
   #yaml-cpp
   dl
   pthread
   rt
   #/reg/common/package/cpsw/framework-R3-4/x86_64-rhel7-opt/lib/libcpsw.so
   #/reg/common/package/yaml-cpp/yaml-cpp-0.5.3/x86_64-rhel7-gcc48-opt/lib/libyaml-cpp.so
)

#
#  dti_eyescan
#
add_executable(dti_eyescan dti_eyescan.cc)

target_link_libraries(dti_eyescan
   service
   cphw
   ${CPSW_LIBRARY}
   ${YAML_CPP_LIBRARY}
   #cpsw
   #yaml-cpp
   dl
   pthread
   rt
   #/reg/common/package/cpsw/framework-R3-4/x86_64-rhel7-opt/lib/libcpsw.so
   #/reg/common/package/yaml-cpp/yaml-cpp-0.5.3/x86_64-rhel7-gcc48-opt/lib/libyaml-cpp.so
)

#
#  dtiHsrCfg
#
add_executable(dtiHsrCfg dtiHsrCfg.cc)

target_link_libraries(dtiHsrCfg
   service
   cphw
   dti
   /reg/common/package/epicsca/3.14.12/lib/x86_64-rhel7-opt/libca.so
   epicstools
   ${CPSW_LIBRARY}
   ${YAML_CPP_LIBRARY}
   dl
   pthread
   rt
   #cpsw
   #yaml-cpp
   #/reg/common/package/cpsw/framework-R3-4/x86_64-rhel7-opt/lib/libcpsw.so
   #/reg/common/package/yaml-cpp/yaml-cpp-0.5.3/x86_64-rhel7-gcc48-opt/lib/libyaml-cpp.so
)
