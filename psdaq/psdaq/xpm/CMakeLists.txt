
add_library(xpm SHARED
    PVStats.cc
    PVPStats.cc
    PVCtrls.cc
    PVPCtrls.cc
    PVLinkCtrls.cc
    user_sequence.cc
    XpmSequenceEngine.cc
    Module.cc
)

target_include_directories(xpm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/epics/include
    $ENV{CONDA_PREFIX}/epics/include/os/Linux
)

find_library(YAML_CPP_LIB yaml-cpp PATH $ENV{CONDA_PREFIX}/lib)
find_library(CPSW_LIB cpsw PATH $ENV{CONDA_PREFIX}/lib)
find_library(EPICS_CA_LIB cpsw PATH $ENV{CONDA_PREFIX}/lib)

#
#  xpm_simple
#
add_executable(xpm_simple xpm_simple.cc)

target_link_libraries(xpm_simple
   service
   cphw
   ${CPSW_LIB}
   ${YAML_CPP_LIB}
   dl
   pthread
   rt
)

#
#  xpmPVs
#
add_executable(xpmPVs xpmPVs.cc)

target_link_libraries(xpmPVs
   service
   cphw
   epicstools
   xpm
   ${EPICS_CA_LIB}
   ${CPSW_LIB}
   ${YAML_CPP_LIB}
   dl
   pthread
   rt
)

#
#  xpmPeek
#
add_executable(xpmPeek xpmPeek.cc)

target_link_libraries(xpmPeek
   service
   cphwr
   xpm
   ${EPICS_CA_LIB}
   epicstools
   dl
   pthread
   rt
)

#
#  xpmHsrCfg
#
add_executable(xpmHsrCfg xpmHsrCfg.cc)

target_link_libraries(xpmHsrCfg
   service
   cphwr
   xpm
   ${EPICS_CA_LIB}
   epicstools
   ${CPSW_LIB}
   ${YAML_CPP_LIB}
   dl
   pthread
   rt
)

install(TARGETS xpm
		xpm_simple
		xpmPVs
		xpmPeek
		xpmHsrCfg
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
