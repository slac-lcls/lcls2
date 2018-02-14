add_library(app SHARED
    AppUtils.cc
)

#
#
#
add_executable(h5writer
    h5writer.cc
)
target_link_libraries(h5writer
    pshdf5
    xtcdata
)

#
#
#
add_executable(hpsBldServer hpsBldServer.cc)

target_include_directories(hpsBldServer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)

find_library(YAML_CPP_LIB yaml-cpp $ENV{CONDA_PREFIX}/lib)
find_library(CPSW_LIB cpsw $ENV{CONDA_PREFIX}/lib)
find_library(EPICS_CA_LIB cpsw PATH $ENV{CONDA_PREFIX}/lib)

target_link_libraries(hpsBldServer
   cphw
   bld
   ${CPSW_LIB}
   ${YAML_CPP_LIB}
   dl
   pthread
   rt
)

add_executable(hpsBldClient hpsBldClient.cc)

target_include_directories(hpsBldClient PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)

target_link_libraries(hpsBldClient
   cphw
   bld
   tpr
   ${CPSW_LIB}
   ${YAML_CPP_LIB}
   app
   dl
   pthread
   rt
)

#
#  tprca
#
add_executable(tprca tprca.cc)

target_include_directories(tprca PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)

target_link_libraries(tprca
   epicstools
   ${EPICS_CA_LIB}
#   ${CPSW_LIB}
#   ${YAML_CPP_LIB}
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
    $ENV{CONDA_PREFIX}/include
)

target_link_libraries(tprsh
   epicstools
   ${EPICS_CA_LIB}
#   ${CPSW_LIB}
#   ${YAML_CPP_LIB}
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
    $ENV{CONDA_PREFIX}/include
)

target_link_libraries(tprutil
   tpr
   dl
   pthread
   rt
)

install(TARGETS app
		h5writer
		hpsBldServer
		hpsBldClient
		tprca
		tprsh
		tprutil
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
