add_library(constfracdiscrim SHARED
    src/ConstFracDiscrim.cc
)

target_link_libraries(constfracdiscrim
)


target_include_directories(constfracdiscrim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
