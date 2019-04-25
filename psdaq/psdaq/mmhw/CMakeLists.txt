
add_library(mmhw SHARED
    AxiVersion.cc
    HexFile.cc
    McsFile.cc
    RegProxy.cc
    RingBuffer.cc
    GthEyeScan.cc
    Xvc.cc
)

target_include_directories(mmhw PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

add_executable(mmregtest mmregtest.cc)

target_include_directories(mmregtest PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

install(TARGETS mmhw
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
