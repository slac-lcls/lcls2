
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

#
#  dti_simple
#
add_executable(dti_simple dti_simple.cc)

target_link_libraries(dti_simple
   dl
   pthread
   service
   cphw
   /reg/common/package/cpsw/framework-R3-4/x86_64-rhel7-opt/lib/libcpsw.so
   /reg/common/package/yaml-cpp/yaml-cpp-0.5.3/x86_64-rhel7-gcc48-opt/lib/libyaml-cpp.so
)

#
#  dtiPVs
#
add_executable(dtiPVs dtiPVs.cc)

target_link_libraries(dtiPVs
   dl
   pthread
   service
   cphw
   epicstools
   dti
   /reg/common/package/epicsca/3.14.12/lib/x86_64-rhel7-opt/libca.so
   /reg/common/package/cpsw/framework-R3-4/x86_64-rhel7-opt/lib/libcpsw.so
   /reg/common/package/yaml-cpp/yaml-cpp-0.5.3/x86_64-rhel7-gcc48-opt/lib/libyaml-cpp.so
)

#
#  dti_eyescan
#
add_executable(dti_eyescan dti_eyescan.cc)

target_link_libraries(dti_eyescan
   dl
   pthread
   service
   cphw
   /reg/common/package/cpsw/framework-R3-4/x86_64-rhel7-opt/lib/libcpsw.so
   /reg/common/package/yaml-cpp/yaml-cpp-0.5.3/x86_64-rhel7-gcc48-opt/lib/libyaml-cpp.so
)

#
#  dtiHsrCfg
#
add_executable(dtiHsrCfg dtiHsrCfg.cc)

target_link_libraries(dtiHsrCfg
   dl
   pthread
   service
   cphw
   epicstools
   dti
   /reg/common/package/epicsca/3.14.12/lib/x86_64-rhel7-opt/libca.so
   /reg/common/package/cpsw/framework-R3-4/x86_64-rhel7-opt/lib/libcpsw.so
   /reg/common/package/yaml-cpp/yaml-cpp-0.5.3/x86_64-rhel7-gcc48-opt/lib/libyaml-cpp.so
)
