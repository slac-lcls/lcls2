
#add_library(dti SHARED
#    PVStats.cc
#    PVCtrls.cc
#    Module.cc
#)

#target_include_directories(dti PUBLIC
#    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
#    /reg/common/package/epicsca/3.14.12/include
#    /reg/common/package/epicsca/3.14.12/include/os/Linux
#)

#
#  hsd_eyescan
#
add_executable(hsd_eyescan hsd_eyescan.cc)

target_link_libraries(hsd_eyescan
   dl
   pthread
   service
   mmhw
)

#
#  hsdRead
#
add_executable(hsdRead hsdRead.cc)

target_include_directories(hsdRead PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_link_libraries(hsdRead
   pthread
)
