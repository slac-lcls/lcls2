
add_library(mmhw SHARED
    AxiVersion.cc
    RingBuffer.cc
    GthEyeScan.cc
)

target_include_directories(mmhw PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

install(TARGETS mmhw
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
