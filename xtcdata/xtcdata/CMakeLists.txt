add_subdirectory(xtc)
add_subdirectory(app)

install(EXPORT xtcdataTargets
    FILE xtcdataConfig.cmake
    DESTINATION lib/cmake/xtcdata
    NAMESPACE xtcdata::
)
