
#
#  xStatus
#
add_executable(xStatus xStatus.cpp)

add_executable(xLoopTest xLoopTest.cpp)

add_executable(xRead xRead.cpp)

target_link_libraries(xRead
   pthread
   rt
   ${EPICS_CA_LIB}
   ${EPICS_PVDATA_LIB}
   ${EPICS_PVACCESS_LIB}
   ${EPICS_PVACLIENT_LIB}
)

add_executable(xMargin xMargin.cpp)

target_link_libraries(xMargin
   pthread
)

add_executable(xError xError.cpp)

target_link_libraries(xError
   pthread
)
