add_library(app SHARED
    AppUtils.cc
)

add_executable(hpsBldServer hpsBldServer.cc)

target_include_directories(hpsBldServer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(hpsBldServer
   app
   cphw
   bld
   epicstools
   cpsw::cpsw
   ${YAML_CPP_LIB}
   dl
   pthread
   rt
)

add_executable(hpsBldClient hpsBldClient.cc)

target_include_directories(hpsBldClient PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(hpsBldClient
   app
   kcu
   cphw
   bld
   epicstools
   xtcdata::xtc
   cpsw::cpsw
   ${YAML_CPP_LIB}
   dl
   pthread
   rt
)


#
#  tprioc
#
add_executable(tprioc tprca.cc)

target_include_directories(tprioc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(tprioc
   epicstools
   tpr
   app
   dl
   pthread
   rt
)

#
#  tprsh
#
add_executable(tprsh tprsh.cc)

target_include_directories(tprsh PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(tprsh
   epicstools
   app
   dl
   pthread
   rt
)

#
#  tprutil
#
add_executable(tprutil tprutil.cc)

target_include_directories(tprutil PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(tprutil
   tpr
   dl
   pthread
   rt
)

#
#  pgpca
#
add_executable(pgpca pgpca.cc)

target_include_directories(pgpca PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(pgpca
   service
   epicstools
   dl
   pthread
   rt
)

install(TARGETS app
		hpsBldServer
		hpsBldClient
		tprioc
		tprsh
		tprutil
        pgpca
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
