
add_library(xpm SHARED
    PVSeq.cc
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
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(xpm
    epicstools
    cpsw::cpsw
    ${YAML_CPP_LIB}
    xtcdata::xtc
)

#
#  xpm_simple
#
add_executable(xpm_simple xpm_simple.cc)

target_link_libraries(xpm_simple
   service
   cphw
   cpsw::cpsw
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
   xpm
   app
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
   app
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
   app
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
