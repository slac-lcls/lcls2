set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(xpm
    epicstools
    ${YAML_CPP_LIB}
    xtc
)

#
#  xpm_simple
#
add_executable(xpm_simple xpm_simple.cc)

target_link_libraries(xpm_simple
   service
   cphw
   ${YAML_CPP_LIB}
   dl
   Threads::Threads
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
   Threads::Threads
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
   Threads::Threads
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
   Threads::Threads
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
