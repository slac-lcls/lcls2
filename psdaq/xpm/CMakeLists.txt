
add_library(xpm SHARED
    PVStats.cc
    PVPStats.cc
    PVCtrls.cc
    PVPCtrls.cc
    PVLinkCtrls.cc
    Module.cc
)

target_include_directories(xpm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    /reg/common/package/epicsca/3.14.12/include
    /reg/common/package/epicsca/3.14.12/include/os/Linux
)

#
#  xpm_simple
#
add_executable(xpm_simple xpm_simple.cc)

target_link_libraries(xpm_simple
   dl
   pthread
   service
   cphw
   /reg/common/package/cpsw/framework-R3-4/x86_64-rhel7-opt/lib/libcpsw.so
   /reg/common/package/yaml-cpp/yaml-cpp-0.5.3/x86_64-rhel7-gcc48-opt/lib/libyaml-cpp.so
)

#
#  xpmPVs
#
add_executable(xpmPVs xpmPVs.cc)

target_link_libraries(xpmPVs
   dl
   pthread
   service
   cphw
   epicstools
   xpm
   /reg/common/package/epicsca/3.14.12/lib/x86_64-rhel7-opt/libca.so
   /reg/common/package/cpsw/framework-R3-4/x86_64-rhel7-opt/lib/libcpsw.so
   /reg/common/package/yaml-cpp/yaml-cpp-0.5.3/x86_64-rhel7-gcc48-opt/lib/libyaml-cpp.so
)
