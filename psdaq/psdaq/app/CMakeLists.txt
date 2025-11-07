find_package(PythonLibs REQUIRED)
find_package(nlohmann_json REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(app SHARED
    AppUtils.cc
)

add_executable(hpsBldServer hpsBldServer.cc)

target_include_directories(hpsBldServer PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    ${Readline_INCLUDE_DIR}
)

target_link_libraries(hpsBldServer
   app
   bld
   epicstools
   dl
   ${Readline_LIBRARY}
   Threads::Threads
   rt
   ${PYTHON_LIBRARIES}
)

add_executable(hpsBldClient hpsBldClient.cc)

target_include_directories(hpsBldClient PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    ${Readline_INCLUDE_DIR}
)

target_link_libraries(hpsBldClient
   app
   kcu
   bld
   epicstools
   xtc
   dl
   ${Readline_LIBRARY}
   Threads::Threads
   rt
)

add_executable(hpsBldCopy hpsBldCopy.cc)

target_include_directories(hpsBldCopy PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    ${Readline_INCLUDE_DIR}
)

target_link_libraries(hpsBldCopy
   app
   bld
   epicstools
   xtc
   dl
   ${Readline_LIBRARY}
   Threads::Threads
   rt
)

add_executable(lclsBldServer lclsBldServer.cc)

target_include_directories(lclsBldServer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    ${Readline_INCLUDE_DIR}
)

target_link_libraries(lclsBldServer
   app
   tpr
   dl
   ${Readline_LIBRARY}
   Threads::Threads
   rt
)

add_executable(tprtrig tprtrig.cc)

target_include_directories(tprtrig PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(tprtrig
   app
   tpr
   dl
   ${Readline_LIBRARY}
   Threads::Threads
   rt
)

#
#  tprioc
#
add_executable(tprioc tprca.cc)

target_include_directories(tprioc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    ${Readline_INCLUDE_DIR}
)

target_link_libraries(tprioc
   epicstools
   tpr
   app
   dl
   ${Readline_LIBRARY}
   Threads::Threads
   rt
)

#
#  tprsh
#
add_executable(tprsh tprsh.cc)

target_include_directories(tprsh PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    ${Readline_INCLUDE_DIR}
)

target_link_libraries(tprsh
   epicstools
   app
   dl
   ${Readline_LIBRARY}
   Threads::Threads
   rt
)

#
#  tprutil
#
add_executable(tprutil tprutil.cc)

target_include_directories(tprutil PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    ${Readline_INCLUDE_DIR}
)

target_link_libraries(tprutil
   tpr
   dl
   ${Readline_LIBRARY}
   Threads::Threads
   rt
)

#
#  xcasttest
#
add_executable(xcasttest xcasttest.cc)

target_include_directories(xcasttest PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    ${Readline_INCLUDE_DIR}
)

target_link_libraries(xcasttest
   app
   dl
   ${Readline_LIBRARY}
   Threads::Threads
   rt
)

#
#  pgpca
#
add_executable(pgpca pgpca.cc)

target_include_directories(pgpca PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    ${Readline_INCLUDE_DIR}
)

target_link_libraries(pgpca
   service
   epicstools
   dl
   ${Readline_LIBRARY}
   Threads::Threads
   rt
)

#
#  pvcam
#
add_executable(pvcam pvcam.cc)

target_include_directories(pvcam PUBLIC
    pvcam
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    ${Readline_INCLUDE_DIR}
)

target_link_libraries(pvcam
   epicstools
   ${EPICS_PVACCESS_LIB}
   ${EPICS_PVDATA_LIB}
   ${EPICS_PVACLIENT_LIB}
   ${EPICS_CA_LIB}
   tpr
   ${READLINE_LIBRARY}
   dl
   rt
)

#
#  bldpayload
#
add_executable(bldpayload bldpayload.cc)

target_include_directories(bldpayload PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $ENV{CONDA_PREFIX}/include
)

target_link_libraries(bldpayload
   xtc
   epicstools
   ${EPICS_PVACCESS_LIB}
   ${EPICS_PVDATA_LIB}
   ${EPICS_PVACLIENT_LIB}
   ${EPICS_CA_LIB}
   ${READLINE_LIBRARY}
   dl
   rt
)

#
#  sim_udpencoder
#
add_executable(sim_udpencoder sim_udpencoder.cc)

target_include_directories(sim_udpencoder PUBLIC
    sim_udpencoder
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../psalg>
    ${ZeroMQ_INCLUDE_DIRS}
    ${Readline_INCLUDE_DIR}
    ${RapidJSON_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
)

target_link_libraries(sim_udpencoder
   tpr
   libfabric::fabric
   prometheus-cpp::push
   nlohmann_json::nlohmann_json
   ${PYTHON_LIBRARIES}
   ${Readline_LIBRARY}
   dl
   rt
)

#
#  json2xtc
#
add_executable(json2xtc
    json2xtc.cc
)

target_link_libraries(json2xtc
    xtc
    service
    ${Readline_LIBRARY}
    Threads::Threads
    rt
)

target_include_directories(json2xtc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    ${RapidJSON_INCLUDE_DIRS}
    ${Readline_INCLUDE_DIR}
)

install(TARGETS app
#               bldpayload
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
                sim_udpencoder
                json2xtc
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
