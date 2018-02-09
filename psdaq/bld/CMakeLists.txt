
add_library(bld SHARED
    HpsEventIterator.cc
    BldServer.cc
    templates.cc
)

target_include_directories(bld PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)

find_library(YAML_CPP_LIB yaml-cpp $ENV{CONDA_PREFIX}/lib)
find_library(CPSW_LIB cpsw $ENV{CONDA_PREFIX}/lib)

add_executable(hpsBldServer hpsBldServer.cc)

target_include_directories(hpsBldServer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $ENV{CONDA_PREFIX}/include
)

target_link_libraries(hpsBldServer
   cphw
   bld
   ${CPSW_LIB}
   ${YAML_CPP_LIB}
   dl
   pthread
   rt
)
