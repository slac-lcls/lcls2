
add_library(dgram SHARED
    dgram.cc
)

target_link_libraries(dgram
    xtc
    ${PYTHON_LIBRARIES}
)

if(PSANA_USE_LEGION)
  target_compile_definitions(dgram PRIVATE PSANA_USE_LEGION)
  target_link_libraries(dgram Legion::Legion)
endif()

target_include_directories(dgram PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> 
    ${PYTHON_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIRS}
)

set_target_properties(dgram PROPERTIES PREFIX "")

add_test(NAME xtcwriter COMMAND ${CMAKE_BINARY_DIR}/xtcdata/xtcwriter
	 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
add_test(NAME testxtc COMMAND python testxtc.py
	 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
add_test(NAME h5writer COMMAND ${CMAKE_BINARY_DIR}/psdaq/app/h5writer
	 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
add_test(NAME testh5 COMMAND python testh5.py
	 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
add_test(NAME testdet COMMAND python testdet.py
	 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
