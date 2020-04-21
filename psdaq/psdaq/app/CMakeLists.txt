find_package(PythonLibs REQUIRED)

add_library(app SHARED
    AppUtils.cc
)

add_executable(hpsBldServer hpsBldServer.cc)

target_include_directories(hpsBldServer PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(hpsBldServer
   app
   bld
   epicstools
   dl
   pthread
   rt
   ${PYTHON_LIBRARIES}
)

add_executable(hpsBldClient hpsBldClient.cc)

target_include_directories(hpsBldClient PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(hpsBldClient
   app
   kcu
   bld
   epicstools
   xtcdata::xtc
   dl
   pthread
   rt
)

add_executable(hpsBldCopy hpsBldCopy.cc)

target_include_directories(hpsBldCopy PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(hpsBldCopy
   app
   bld
   epicstools
   xtcdata::xtc
   dl
   pthread
   rt
)

add_executable(lclsBldServer lclsBldServer.cc)

target_include_directories(lclsBldServer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(lclsBldServer
   app
   tpr
   dl
   pthread
   rt
)

add_executable(tprtrig tprtrig.cc)

target_include_directories(tprtrig PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(tprtrig
   app
   tpr
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
#  xcasttest
#
add_executable(xcasttest xcasttest.cc)

target_include_directories(xcasttest PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(xcasttest
   app
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

#
#  pvcam
#
add_executable(pvcam pvcam.cc)

target_include_directories(pvcam PUBLIC
    pvcam
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)

add_executable(json2xtc
    json2xtc.cc
)

target_link_libraries(json2xtc
    xtcdata::xtc
    service
    pthread
)

target_include_directories(json2xtc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${RAPIDJSON_INCLUDE_DIRS}
)

target_link_libraries(pvcam
   epicstools
   ${EPICS_PVACCESS_LIB}
   ${EPICS_PVDATA_LIB}
   ${EPICS_PVACLIENT_LIB}
   ${EPICS_CA_LIB}
   tpr
   dl
   rt
)


install(TARGETS app
		hpsBldServer
		hpsBldClient
		hpsBldCopy
                lclsBldServer
                tprtrig
		tprioc
		tprsh
                tprutil
		xcasttest
                pgpca
                pvcam
                json2xtc
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
