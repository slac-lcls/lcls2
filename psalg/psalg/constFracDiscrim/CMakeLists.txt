add_library(constfracdiscrim SHARED
    src/ConstFracDiscrim.cc
)

#find_package(GSL REQUIRED)

target_link_libraries(constfracdiscrim
  #GSL::gsl GSL::gslcblas m
)


target_include_directories(constfracdiscrim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES
    ConstFracDiscrim.hh
    DESTINATION include/psalg/constFracDiscrim
)

install(TARGETS constfracdiscrim
    EXPORT psalgTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
