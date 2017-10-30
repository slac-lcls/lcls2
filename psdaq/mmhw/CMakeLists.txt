
add_library(mmhw SHARED
    AxiVersion.cc
    RingBuffer.cc
    GthEyeScan.cc
)

target_include_directories(mmhw PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

